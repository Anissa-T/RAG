import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

torch.cuda.empty_cache()

""" 
Charger des documents texte depuis un dossier
Cette fonction charge récursivement tous les fichiers .txt dans le dossier spécifié.
"""
def load_documents(directory_path):
    try:
        loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

""" 
Découper les documents en petits fragments plus faciles à traiter
La stratégie récursive permet de préserver le contexte.
"""
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

""" 
Créer des embeddings et indexer les fragments avec FAISS pour une recherche rapide
"""
def create_vectorstore(chunks, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"Created FAISS vectorstore with {len(chunks)} documents")
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise

""" 
Sauvegarder l'index FAISS pour éviter de recalculer à chaque exécution
"""
def save_vectorstore(vectorstore, path="faiss_index"):
    try:
        vectorstore.save_local(path)
        print(f"Saved FAISS index to {path}")
    except Exception as e:
        print(f"Error saving vectorstore: {e}")

""" 
Charger un index FAISS déjà existant depuis le disque
"""
def load_vectorstore(path="faiss_index", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS index from {path}")
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise

""" 
Charger un modèle de langage léger compatible CPU pour la génération de texte
"""
def load_cpu_friendly_llm(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    try:
        print(f"Loading language model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print("Language model loaded successfully")
        return llm
    except Exception as e:
        print(f"Error loading language model: {e}")
        raise
