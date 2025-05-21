from transformers import pipeline
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.evaluate import evaluate
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

loader = TextLoader("./data/") #METTEZ VOTRE DOCUMENT
data = loader.load()
documents = ## TRAITEZ VOS DONNEES

generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_synthetic_qa(context):
    q_prompt = f"Generate a question that can be answered by this text:\n{context}"
    a_prompt = f"Answer the following question based on the text:\n{context}"

    question = generator(q_prompt, max_new_tokens=50)[0]["generated_text"]
    answer = generator(f"{a_prompt}\nQuestion: {question}", max_new_tokens=80)[0]["generated_text"]

    return question.strip(), answer.strip()

samples = []
for doc in documents[5:6]:
    question, reference_answer = generate_synthetic_qa(doc)
    samples.append(LLMTestCase(
        input=question,
        context=[doc],
        expected_output=reference_answer,       # Réponse qui a été générée
        actual_output = #METTEZ ICI LA SORTIE DE VOTRE RAG
    ))

synthetic_dataset = EvaluationDataset(test_cases=samples)

class EmbeddingSimilarityMetric(BaseMetric):
    def __init__(self, threshold=0.75, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    def measure(self, test_case: LLMTestCase):
        context = " ".join(test_case.context)
        embeddings = self.model.encode([test_case.actual_output, context])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        test_case.metric_scores[self.name()] = float(score)
        return score

    def is_pass(self, score: float) -> bool:
        return score >= self.threshold

    def name(self):
        return "embedding_similarity"

    def rationale(self) -> str:
        return "Scores based on cosine similarity between context and generated answer"

metric = EmbeddingSimilarityMetric(threshold=0.75)
results = evaluate(dataset=synthetic_dataset, metrics=[metric])

print("Similarity Score:", results[0].metric_scores["embedding_similarity"])



# contex 



# prend un paragraphe, genere une question et cherche dans ce paragraphe la reponse
# puis compare la reponse avec la reponse attendue  
# et renvoie un score de similarite entre 0 et 1
# 0 = pas de similarite, 1 = reponse identique
# 0.5 = reponse similaire mais pas identique
# 0.75 = reponse tres similaire mais pas identique
# 0.9 = reponse presque identique mais pas identique
# 0.95 = reponse identique mais pas exactement identique



# Il faut comparer la reponse generee par rag.py (actual output) avec la reponse de ce code rag_evaluation_to_complete.py (expected output)

# Score rouge
# Score bleu ?