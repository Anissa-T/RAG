import os
import time
try:
    import psutil
except ImportError:
    psutil = None
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
from langdetect import detect

from rag import load_vectorstore, load_cpu_friendly_llm, answer_question_rag
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# --- Configuration ---
INDEX_PATHS = {
    "default": "faiss_index",
    "embed_mpnet": "faiss_index_embed_mpnet"
}
EMBED_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "paraphrase-multilingual-mpnet-base-v2"
}
LLM_MODELS = {
    "Qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}
TOP_K_LIST = [1, 3, 5, 10]
TEST_QUESTIONS = [
    ("Qui est Charles Perrault ?", "Charles Perrault est un écrivain français du XVIIe siècle, auteur de célèbres contes de fées.", "fr"),
    ("Who is Charles Perrault?", "Charles Perrault was a 17th-century French writer, known for famous fairy tales.", "en"),
    ("¿Quién es Charles Perrault?", "Charles Perrault fue un escritor francés del siglo XVII, conocido por sus famosos cuentos de hadas.", "es")
]
OUTPUT_FILE = "answers/evaluation_rag_result.txt"

# Token-level F1 score
def compute_f1(a: str, b: str) -> float:
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    common = set(a_tokens) & set(b_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(a_tokens)
    rec = len(common) / len(b_tokens)
    return 2 * prec * rec / (prec + rec)

class EvaluationRunner:
    def __init__(self):
        os.makedirs("answers", exist_ok=True)
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.embedder = None

    def evaluate_scenario(self, desc, index_path, embed_model, llm_model, top_k, lang_hint):
        vs = load_vectorstore(index_path, embed_model)
        text_gen = load_cpu_friendly_llm(llm_model)
        if self.embedder is None or getattr(self.embedder, 'model_name', None) != embed_model:
            self.embedder = SentenceTransformer(embed_model)

        test_cases = []
        for q, ref, lang in TEST_QUESTIONS:
            # measure latency and memory
            start = time.perf_counter()
            mem_before = psutil.Process().memory_info().rss if psutil else None
            pred, docs = answer_question_rag(q, vs, text_gen, top_k=top_k, lang_hint=(lang_hint and lang))
            end = time.perf_counter()
            mem_after = psutil.Process().memory_info().rss if psutil else None

            latency = end - start
            mem_usage = (mem_after - mem_before) / (1024**2) if mem_before and mem_after else None

            # retrieval precision: check if expected answer in any context
            context_texts = [d.page_content for d in docs]
            retrieval = any(ref.lower() in txt.lower() for txt in context_texts)

            # exact match & F1
            exact = (pred.strip().lower() == ref.strip().lower())
            f1 = compute_f1(pred, ref)

            # language consistency
            pred_lang = detect(pred) if 'detect' in globals() else None
            lang_consistent = (pred_lang == lang)

            # unknown answer detection
            unknown = (pred.strip().lower() == "je ne sais pas.")

            tc = LLMTestCase(
                input=q,
                context=context_texts,
                expected_output=ref,
                actual_output=pred
            )
            tc.metric_scores = {
                'retrieval_precision': int(retrieval),
                'exact_match': int(exact),
                'f1': f1,
                'bleu': self.bleu.compute(predictions=[pred], references=[ref])["bleu"],
                'rougel': self.rouge.compute(predictions=[pred], references=[ref])["rougeL"],
                'emb_sim': float(cosine_similarity([self.embedder.encode(pred)], [self.embedder.encode(ref)])[0][0]),
                'lang_consistent': int(lang_consistent),
                'unknown': int(unknown),
                'latency_s': latency,
                'mem_delta_mb': mem_usage
            }
            tc.context_language = lang
            test_cases.append(tc)

        # write results
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"=== Scenario: {desc} ===\n")
            for tc in test_cases:
                f.write(f"Question ({tc.context_language}): {tc.input}\n")
                f.write(f"Ref     : {tc.expected_output}\n")
                f.write(f"Pred    : {tc.actual_output}\n")
                for k, v in tc.metric_scores.items():
                    f.write(f"{k}: {v}\n")
                f.write("-"*40 + "\n")
            f.write("\n")

    def run(self):
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        # tests top-k
        for k in TOP_K_LIST:
            self.evaluate_scenario(f"top_k={k}", INDEX_PATHS['default'], EMBED_MODELS['MiniLM'], LLM_MODELS['Qwen'], top_k=k, lang_hint=True)
        # embeddings comparison
        for name, em in EMBED_MODELS.items():
            idx = 'default' if name=='MiniLM' else 'embed_mpnet'
            self.evaluate_scenario(f"embed={name}", INDEX_PATHS[idx], em, LLM_MODELS['Qwen'], top_k=3, lang_hint=True)
        # LLM comparison
        for name, lm in LLM_MODELS.items():
            self.evaluate_scenario(f"llm={name}", INDEX_PATHS['default'], EMBED_MODELS['MiniLM'], lm, top_k=3, lang_hint=True)
        # no filter
        self.evaluate_scenario("no_filter", INDEX_PATHS['default'], EMBED_MODELS['MiniLM'], LLM_MODELS['Qwen'], top_k=3, lang_hint=False)

if __name__ == "__main__":
    runner = EvaluationRunner()
    runner.run()
