import os
from datetime import datetime
import argparse
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Détection automatique de la langue
try:
    from langdetect import detect
except ImportError:
    print("[!] langdetect non installé, la détection automatique sera inactive.")
    def detect(text):
        return None

# Vider le cache GPU si nécessaire
torch.cuda.empty_cache()

# Chargement des documents et ajout de la métadonnée 'lang'
def load_documents(data_path):
    loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"[+] {len(docs)} documents chargés depuis {data_path}")
    for doc in docs:
        src = doc.metadata.get("source", "")
        if "/fr/" in src:
            doc.metadata["lang"] = "fr"
        elif "/en/" in src:
            doc.metadata["lang"] = "en"
        elif "/es/" in src:
            doc.metadata["lang"] = "es"
        else:
            doc.metadata["lang"] = None
    return docs

# Découpage des documents en fragments
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"[+] {len(chunks)} fragments obtenus")
    return chunks

# Création de l'index FAISS
def create_vectorstore(chunks, embedding_model_name):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_documents(chunks, embeddings)
    print("[+] Index FAISS créé")
    return vs

# Sauvegarde de l'index
def save_vectorstore(vectorstore, path):
    vectorstore.save_local(path)
    print(f"[+] Index sauvegardé dans {path}")

# Chargement de l'index
def load_vectorstore(path, embedding_model_name):
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"[+] Index chargé depuis {path}")
    return vs

# Chargement du LLM optimisé CPU
def load_cpu_friendly_llm(model_id):
    print(f"[+] Chargement du LLM {model_id} sur CPU")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15
    )
    print("[+] LLM prêt")
    return text_gen

# Détection de la langue
def detect_language(text):
    try:
        return detect(text)
    except:
        return None

# Question-réponse retrieval
def answer_question_rag(question, vectorstore, text_gen, top_k=3, lang_hint=None):
    lang = lang_hint or detect_language(question)
    filt = {"lang": lang} if lang else None
    docs = vectorstore.similarity_search(question, k=top_k, filter=filt)
    context = "\n".join(d.page_content for d in docs)

    lang_names = {"en": "anglais", "fr": "français", "es": "espagnol"}
    lang_full = lang_names.get(lang, "la langue du contexte")

    prompt = (
        f"Contexte (en {lang_full}):\n{context}\n\n"
        f"Question: {question}\n"
        f"Répondez en {lang_full} uniquement selon le contexte."
        f" Si l'info manque, répondez \"Je ne sais pas.\"\nRéponse:"
    )
    out = text_gen(prompt)
    answer = out[0]["generated_text"].strip()
    return answer, docs

# Point d'entrée du script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Système RAG pour contes multilingues"
    )
    parser.add_argument(
        "--data-path",
        default="data",
        help="Dossier des fichiers .txt"
    )
    parser.add_argument(
        "--index-path",
        default="faiss_index",
        help="Dossier index FAISS"
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Reconstruire l'index FAISS"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Modèle d'embeddings"
    )
    parser.add_argument(
        "--llm-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Modèle LLM"
    )
    parser.add_argument(
        "--question",
        help="Question à poser au système RAG"
    )
    parser.add_argument(
        "--lang",
        help="Langue de la question (fr/en/es)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de contexts à récupérer"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Taille des fragments"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chevauchement des fragments"
    )

    args = parser.parse_args()

    if not args.build_index and not args.question:
        parser.error("--question requis si pas de --build-index")

    if args.build_index:
        docs = load_documents(args.data_path)
        chunks = split_documents(docs, args.chunk_size, args.chunk_overlap)
        vs = create_vectorstore(chunks, args.embedding_model)
        save_vectorstore(vs, args.index_path)
    else:
        vs = load_vectorstore(args.index_path, args.embedding_model)
        text_gen = load_cpu_friendly_llm(args.llm_model)
        answer, docs_used = answer_question_rag(
            args.question, vs, text_gen,
            top_k=args.top_k, lang_hint=args.lang
        )
        print(f"Réponse : {answer}")
        os.makedirs("answers", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"answers/answer_{ts}.txt"
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"Question: {args.question}\n")
            f.write(f"Langue: {args.lang or detect_language(args.question)}\n\n")
            for i, d in enumerate(docs_used, 1):
                f.write(f"--- Contexte {i}:\n{d.page_content}\n\n")
            f.write(f"Réponse:\n{answer}\n")
        print(f"[+] Sauvegardé dans {fn}")
