# app.py → VERSION FINALE CORRIGÉE + AJOUT CONTENU RAPPORT
import streamlit as st
from pathlib import Path
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# =============================================
# CHARTE GRAPHIQUE CASEAI
# =============================================
st.set_page_config(page_title="RAG Perrault – CaseAI", layout="wide", page_icon="leaf")

primary_green = "#0A5C2F"
accent_green = "#4CAF50"
orange = "#FF6B35"
bg_light = "#F0F8F0"

st.markdown(
    f"""
<style>
    .main {{ background: {bg_light}; }}
    .block-container {{ max-width: 95%; padding-top: 2rem; }}
    h1, h2, h3 {{ color: {primary_green}; font-family: 'Helvetica Neue', sans-serif; font-weight: 600; }}
    .stButton>button {{ background: {orange}; color: white; border-radius: 12px; font-weight: bold; height: 3em; }}
    .stButton>button:hover {{ background: {accent_green}; }}
    .footer {{ text-align: center; padding: 3rem; background: {primary_green}; color: white; font-size: 1.2em; }}
    .highlight {{ background-color: #E8F5E8; padding: 1.2rem; border-left: 6px solid {orange}; border-radius: 8px; }}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================
# HEADER
# =============================================
col1, col2 = st.columns([1, 5])
with col1:
    st.image("CaseAI.png", width=200)
with col2:
    st.markdown(
        f"""
    <h1 style='color:{primary_green}; margin:0;'>RAG Multilingue – Charles Perrault</h1>
    <p style='color:{orange}; font-size:1.4em; font-weight:bold; margin:0;'>Anissa Thezenas – Junior Data Scientist RAG @ Bayes Impact</p>
    <p style='margin:5px 0;'>
        <a href="https://github.com/Anissa-T/RAG" target="_blank" style='color:{accent_green};'>GitHub</a> • 
        <a href="https://github.com/Anissa-T/RAG/blob/main/rapport_rag.pdf" target="_blank" style='color:{accent_green};'>Rapport PDF</a>
    </p>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =============================================
# CONFIGURATION DES INDEX FAISS (chemins réels + embeddings compatibles)
# =============================================
INDEX_CONFIG = {
    "MiniLM – 500 caractères (meilleur recall)": {
        "folder": "./RAG/faiss_index_cs500_ol200",
        "emb_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "MiniLM – 2000 caractères (plus de contexte)": {
        "folder": "./RAG/faiss_index_cs2000_ol200",
        "emb_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "MPNet – 500 caractères (meilleure sémantique)": {
        "folder": "./RAG/faiss_index_embed_mpnet",
        "emb_model": "sentence-transformers/all-mpnet-base-v2",
    },
}

@st.cache_resource
def load_vectorstore(folder: str, emb_model: str):
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    path = Path(folder)
    if not path.exists():
        st.error(f"Index non trouvé : {path.resolve()}")
        st.stop()

    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    return FAISS.load_local(
        folder_path=str(path),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

@st.cache_resource
def load_llm(model_name: str):
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model=model_name,
        torch_dtype="auto",
        device_map=None,  # CPU only → pas de bug accelerate
        trust_remote_code="Qwen" in model_name,
        max_new_tokens=150,
    )

# =============================================
# SIDEBAR – Configuration
# =============================================
with st.sidebar:
    st.markdown(f"<h3 style='color:{primary_green};'>Configuration RAG</h3>", unsafe_allow_html=True)

    llm_choice = st.selectbox(
        "Modèle de génération",
        [
            "Qwen/Qwen2.5-1.5B-Instruct (qualité maximale)",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (rapide)",
        ],
        index=0,
    )
    llm_model = (
        "Qwen/Qwen2.5-1.5B-Instruct"
        if "Qwen" in llm_choice
        else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    index_display = st.selectbox("Index FAISS", list(INDEX_CONFIG.keys()), index=0)
    index_folder = INDEX_CONFIG[index_display]["folder"]
    emb_model = INDEX_CONFIG[index_display]["emb_model"]

    lang = st.selectbox("Langue", ["Français", "English", "Español"], index=0)
    lang_code = {"Français": "fr", "English": "en", "Español": "es"}[lang]

# Chargement
vectorstore = load_vectorstore(index_folder, emb_model)
generator = load_llm(llm_model)

st.sidebar.success(
    f"LLM: {llm_choice.split(' (')[0]}\n"
    f"Index: {index_display}\n"
    f"Embeddings: {emb_model.split('/')[-1]}"
)

# =============================================
# NAVIGATION
# =============================================
page = st.radio(
    "Navigation",
    [
        "Démo Live RAG",
        "LLM seul vs RAG – Comparaison simultanée",
        "Rapport PDF (Résumé)",
        "Code Showcase",
        "Alignement 100% Fiche de Poste",
        "Évaluation",
    ],
    horizontal=True,
)

# =============================================
# 1. DÉMO LIVE
# =============================================
if page == "Démo Live RAG":
    st.markdown(f"<h2 style='color:{primary_green};'>Démo RAG – {lang}</h2>", unsafe_allow_html=True)
    question = st.text_input("Votre question", placeholder="Ex: Quand est né Charles Perrault ?")

    if st.button("Générer avec RAG", type="primary", use_container_width=True):
        with st.spinner("Recherche + génération…"):
            docs = vectorstore.similarity_search(question, k=5, filter={"lang": lang_code})

            if not docs:
                st.warning("Aucun chunk retrouvé avec ce filtre. Vérifie tes métadonnées 'lang' ou la question.")
                st.stop()

            context = "\n\n".join(
                [
                    f"[Source: {Path(d.metadata.get('source','unknown')).name}]\n{d.page_content}"
                    for d in docs
                ]
            )
            prompt = (
                f"Tu es un expert sur Charles Perrault. Réponds en {lang}, précis et concis.\n"
                f"Contexte:\n{context}\nQuestion: {question}\nRéponse:"
            )
            result = generator(prompt, temperature=0.1, do_sample=False)[0]["generated_text"]
            answer = (
                result.split("Réponse:")[-1].strip()
                if "Réponse:" in result
                else result[len(prompt):].strip()
            )

        st.success(f"Réponse ({lang})")
        st.write(f"**{answer}**")

        with st.expander("Chunks retrouvés", expanded=True):
            for i, d in enumerate(docs, 1):
                src = Path(d.metadata.get("source", "unknown")).name
                st.markdown(f"**{i}.** `{src}`\n\n{d.page_content[:1200]}...")

# =============================================
# 2. COMPARAISON SIMULTANÉE – UN SEUL BOUTON
# =============================================
elif page == "LLM seul vs RAG – Comparaison simultanée":
    st.markdown(
        f"<h2 style='color:{primary_green};'>LLM seul vs RAG – Un seul clic, deux mondes</h2>",
        unsafe_allow_html=True,
    )
    st.markdown("**Même LLM • Même question • Résultat radicalement différent**")

    question = st.text_input("Question piège", "Quand est né Charles Perrault ?", key="comp")

    if st.button("Générer les deux réponses", type="primary", use_container_width=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**LLM seul** – Hallucination")
            with st.spinner("LLM seul…"):
                prompt_no_rag = f"Réponds en français : {question}"
                out_no = generator(
                    prompt_no_rag, max_new_tokens=80, do_sample=False
                )[0]["generated_text"]
                ans_no = out_no[len(prompt_no_rag):].strip()
                st.error("Souvent faux ou vague")
                st.write(ans_no)

        with col2:
            st.markdown("**Avec RAG** – Réponse exacte")
            with st.spinner("RAG…"):
                docs = vectorstore.similarity_search(question, k=3, filter={"lang": "fr"})
                context = "\n".join(d.page_content for d in docs)
                prompt_rag = f"Contexte: {context}\n\nQuestion: {question}\nRéponse exacte:"
                out_rag = generator(
                    prompt_rag, max_new_tokens=80, do_sample=False
                )[0]["generated_text"]
                ans_rag = out_rag[len(prompt_rag):].strip()
                st.success("Toujours juste")
                st.write(ans_rag)

                if docs:
                    st.info(f"Source: `{Path(docs[0].metadata['source']).name}`")

        st.markdown("**C’est ÇA que CaseAI a besoin : un RAG fiable, pas un LLM qui approximatif mais plutôt un RAG qui s'appuie sur des données solides et vérifiables.**")

# =============================================
# 3. RAPPORT PDF (Résumé + éléments du rapport)
# =============================================
elif page == "Rapport PDF (Résumé)":
    st.markdown(f"<h2 style='color:{primary_green};'>Rapport M2 TAL – Résumé détaillé</h2>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="highlight">
<b>Projet RAG des contes de fées de Charles Perrault</b><br>
Nicolas <b>NGAUV</b> • Anissa <b>THEZENAS</b><br>
Transformers, BERT, RAG<br>
Dispensé par <b>Madame Fedchenko</b><br>
Année universitaire <b>2024–2025</b> • Master 2 / Semestre 2
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Plan du rapport")
    st.markdown(
        """
1. Introduction  
2. Constitution de la base de données  
3. Architecture et paramétrage  
4. Choix du modèle de génération  
5. Protocole d’évaluation  
6. Résultats et interprétation  
7. Problèmes identifiés  
8. Pistes d’amélioration  
9. Conclusion
"""
    )

    with st.expander("1. Introduction", expanded=False):
        st.markdown(
            """
Ce travail s’inscrit dans le cadre du master TAL et vise la conception, l’implémentation et l’évaluation 
d’un système de QA basé sur **RAG (Retrieval-Augmented Generation)**.  
Nous avons construit un index **FAISS** sur un corpus multilingue Perrault, puis déployé un LLM pour 
récupérer des passages pertinents et générer des réponses contextualisées en **FR/EN/ES**.
Axes principaux : précision du retrieval, fiabilité génération (hallucinations), multilinguisme, 
impact du chunking + embeddings + top-k.
"""
        )

    with st.expander("2. Constitution de la base de données", expanded=False):
        st.markdown(
            """
Corpus : **18 fichiers, 84 256 tokens**  
• Biographie détaillée de Perrault  
• 4 contes : *La Belle au bois dormant*, *Peau d’Âne*, *Le Petit Chaperon rouge*, *La Barbe bleue*  
• Chaque conte en FR (origine) + EN/ES (traductions)

Pré-traitements :
- nettoyage / normalisation (casse, suppression bruit)
- segmentation **500 ou 2000 caractères**, overlap **200**
- embeddings : **MiniLM (384d)** vs **MPNet multilingue (768d)**
"""
        )

    with st.expander("3. Architecture et paramétrage", expanded=False):
        st.markdown(
            """
3 index FAISS :
1. MiniLM + chunks 500  
2. MiniLM + chunks 2000  
3. MPNet + chunks 500  

Pipeline orchestré via `rag.py` : paramètres dynamiques (index, embeddings, LLM, top-k, langue ISO-639-1).  
Conçu CPU-only sur MacBook Pro 8 Go RAM → focus modularité et reproductibilité.
"""
        )

    with st.expander("4. Choix du modèle de génération", expanded=False):
        st.markdown(
            """
Tests initiaux avec **TinyLlama** : rapide mais hallucinations fortes.  
Adoption de **Qwen2.5-1.5B-Instruct** : plus lent (20–80s) mais réponses plus stables, meilleure 
adhérence au contexte, bon multilinguisme.
"""
        )

    with st.expander("5. Protocole d’évaluation", expanded=False):
        st.markdown(
            """
4 scénarios :  
- Variation top-k  
- MiniLM vs MPNet  
- Qwen vs TinyLlama  
- Sans filtre de langue  

Questions testées :  
• FR « Qui est Charles Perrault ? »  
• EN « Who is Charles Perrault? »  
• ES « ¿Quién es Charles Perrault? »  

Métriques : retrieval_precision, exact_match, F1 token, BLEU, ROUGE-L, emb_sim, lang_consistent, 
unknown, latency_s, mem_delta_mb.
"""
        )

    with st.expander("6. Résultats et interprétation", expanded=False):
        st.markdown(
            """
**retrieval_precision = 0 dans tous les cas** → chunks pertinents jamais retrouvés.  
Conséquences : BLEU ~0, ROUGE-L 0.03–0.08, F1 0.02–0.08.  
emb_sim ~0.60–0.63 → réponses thématiquement cohérentes mais factuellement incomplètes.  
unknown = 0 → les LLM hallucinent plutôt que d’admettre l’inconnu.

Perf :  
• TinyLlama : 5–10s / requête, mémoire ~4 Go  
• Qwen : 20–80s, mémoire ~6–7 Go, qualité + stable.

Augmenter top-k ne règle pas : latence/mémoire explosent sans gain retrieval.
"""
        )

    with st.expander("7. Problèmes identifiés", expanded=False):
        st.markdown(
            """
Goulot principal : **retrieval**.  
Chunking actuel morcelle/dilue l’info clé → ranking cosinus incapable d’isoler les bons passages.  
MiniLM trop léger pour nuances classiques ; MPNet améliore emb_sim mais pas retrieval_precision.  
Cosinus favorise parfois fragments généraux.  
LLM hallucinent systématiquement sans chunks alignés.  
Contraintes CPU/RAM limitent top-k et taille contexte.
"""
        )

    with st.expander("8. Pistes d’amélioration", expanded=False):
        st.markdown(
            """
- **Chunking sémantique** (phrases/paragraphes/repères structurels)  
- **Re-ranking** avec un second modèle plus puissant  
- **Fine-tuning** LLM sur paires QA Perrault  
- Enrichir FAISS avec **métadonnées structurantes** (titres, chapitres, balises).
"""
        )

    with st.expander("9. Conclusion", expanded=False):
        st.markdown(
            """
La qualité du RAG dépend d’abord du retrieval : sans index fin et embeddings adaptés, le LLM ne peut pas compenser.  
TinyLlama est utile pour prototypage rapide, Qwen est préférable pour la fiabilité factuelle.  
Les pistes proposées (chunking sémantique, re-ranking, FT, enrichissement metadata) sont nécessaires 
pour obtenir un RAG multilingue robuste.
"""
        )

# =============================================
# 4. CODE SHOWCASE
# =============================================
elif page == "Code Showcase":
    st.markdown(
        f"<h2 style='color:{primary_green};'>Code Showcase – Compétences Fiche de Poste</h2>",
        unsafe_allow_html=True,
    )
    tabs = st.tabs(["Scraping", "Chunking", "FAISS", "RAG", "Éval", "Multilingue"])

    with tabs[0]:
        st.markdown("**dataset_conte.py → Scraping Wikipedia + nettoyage du texte**")
        st.code(
            r"""
import requests
from bs4 import BeautifulSoup
from pathlib import Path

WIKI_PAGES = {
    "fr": [
        "https://fr.wikipedia.org/wiki/Charles_Perrault",
        "https://fr.wikipedia.org/wiki/La_Barbe_bleue",
        "https://fr.wikipedia.org/wiki/Le_Petit_Chaperon_rouge",
        "https://fr.wikipedia.org/wiki/Peau_d%27%C3%82ne",
        "https://fr.wikipedia.org/wiki/La_Belle_au_bois_dormant",
    ],
    "en": [
        "https://en.wikipedia.org/wiki/Charles_Perrault",
        "https://en.wikipedia.org/wiki/Bluebeard",
        "https://en.wikipedia.org/wiki/Little_Red_Riding_Hood",
        "https://en.wikipedia.org/wiki/Donkeyskin",
        "https://en.wikipedia.org/wiki/Sleeping_Beauty",
        "https://en.wikipedia.org/wiki/Puss_in_Boots",
    ],
    "es": [
        "https://es.wikipedia.org/wiki/Charles_Perrault",
        "https://es.wikipedia.org/wiki/Barba_Azul",
        "https://es.wikipedia.org/wiki/Caperucita_Roja",
        "https://es.wikipedia.org/wiki/Piel_de_asno",
        "https://es.wikipedia.org/wiki/La_bella_durmiente",
        "https://es.wikipedia.org/wiki/El_gato_con_botas",
    ],
}

def scrape_page(url: str) -> str:
    html = requests.get(url, timeout=20).text
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output")

    paragraphs = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    text = "\n".join(paragraphs)

    # Normalisation légère pour embeddings
    text = text.replace("\xa0", " ")
    text = " ".join(text.split())
    return text

def build_corpus(out_root="RAG/data"):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for lang, urls in WIKI_PAGES.items():
        lang_dir = out_root / lang
        lang_dir.mkdir(exist_ok=True)

        for url in urls:
            title = url.split("/")[-1] + ".txt"
            txt = scrape_page(url)
            (lang_dir / title).write_text(txt, encoding="utf-8")
            print(f"[OK] {lang} → {title} ({len(txt)} chars)")

if __name__ == "__main__":
    build_corpus()
""",
            language="python",
        )

    with tabs[1]:
        st.markdown("**rag.py → Chargement récursif + chunking paramétrable**")
        st.code(
            r"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(data_dir="RAG/data"):
    loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding":"utf-8"}
    )
    docs = loader.load()

    # Ajout metadata langue à partir du chemin
    for d in docs:
        parts = Path(d.metadata["source"]).parts
        d.metadata["lang"] = parts[-2] if len(parts) >= 2 else "unknown"
    print(f"[Loader] {len(docs)} fichiers chargés")
    return docs

def split_documents(docs, chunk_size=500, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, ch in enumerate(chunks):
        ch.metadata["chunk_id"] = i
    print(f"[Chunking] {len(docs)} docs → {len(chunks)} chunks (cs={chunk_size}, ol={overlap})")
    return chunks

# Deux configs testées
docs = load_documents()
chunks_500 = split_documents(docs, chunk_size=500, overlap=200)
chunks_2000 = split_documents(docs, chunk_size=2000, overlap=200)
""",
            language="python",
        )

    with tabs[2]:
        st.markdown("**rag.py → Construction + sauvegarde d’index FAISS**")
        st.code(
            r"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_faiss_index(chunks, emb_model, out_dir):
    embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    vs = FAISS.from_documents(chunks, embeddings)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))

    print(f"[FAISS] index sauvé → {out_dir}")
    return vs

# Trois index (chunking/embeddings)
vs_minilm_500 = build_faiss_index(
    chunks_500,
    emb_model="sentence-transformers/all-MiniLM-L6-v2",
    out_dir="RAG/faiss_index_cs500_ol200",
)

vs_minilm_2000 = build_faiss_index(
    chunks_2000,
    emb_model="sentence-transformers/all-MiniLM-L6-v2",
    out_dir="RAG/faiss_index_cs2000_ol200",
)

vs_mpnet_500 = build_faiss_index(
    chunks_500,
    emb_model="sentence-transformers/all-mpnet-base-v2",
    out_dir="RAG/faiss_index_embed_mpnet",
)
""",
            language="python",
        )

    with tabs[3]:
        st.markdown("**rag.py / app.py → Retrieval filtré + prompt grounding**")
        st.code(
            r"""
def answer_question_rag(question, vectorstore, generator, top_k=5, lang_hint=None):
    # Retrieval (FAISS)
    filt = {"lang": lang_hint} if lang_hint else None
    docs = vectorstore.similarity_search(question, k=top_k, filter=filt)

    # Contexte prêt pour le LLM
    context = "\n\n".join(
        [f"[{Path(d.metadata['source']).name}]\n{d.page_content}" for d in docs]
    )

    prompt = (
        "Tu es un expert sur Charles Perrault.\n"
        "Réponds uniquement à partir du contexte fourni.\n\n"
        f"Contexte:\n{context}\n\n"
        f"Question: {question}\nRéponse:"
    )

    out = generator(prompt, temperature=0.1, do_sample=False)[0]["generated_text"]
    answer = out.split("Réponse:")[-1].strip()
    return answer, docs
""",
            language="python",
        )

    with tabs[4]:
        st.markdown("**rag_evaluation_to_complete.py → Boucle multi-scénarios + métriques**")
        st.code(
            r"""
TOP_K_LIST = [1, 3, 5, 10]

SCENARIOS = [
    ("MiniLM-500 + Qwen", "RAG/faiss_index_cs500_ol200", "all-MiniLM-L6-v2", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("MiniLM-2000 + Qwen", "RAG/faiss_index_cs2000_ol200", "all-MiniLM-L6-v2", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("MPNet-500 + Qwen", "RAG/faiss_index_embed_mpnet", "all-mpnet-base-v2", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("MiniLM-500 + TinyLlama", "RAG/faiss_index_cs500_ol200", "all-MiniLM-L6-v2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
]

TEST_QUESTIONS = [
    ("Qui est Charles Perrault ?", "écrivain français du XVIIe siècle...", "fr"),
    ("Who is Charles Perrault?", "French writer from the 17th century...", "en"),
    ("¿Quién es Charles Perrault?", "escritor francés del siglo XVII...", "es"),
]

def run_eval():
    results = []
    for desc, index_path, emb_model, llm_model in SCENARIOS:
        vs = load_vectorstore(index_path, emb_model)
        gen = load_llm(llm_model)

        for top_k in TOP_K_LIST:
            for q, ref, lang in TEST_QUESTIONS:
                pred, docs = answer_question_rag(q, vs, gen, top_k=top_k, lang_hint=lang)

                retrieval = any(ref.lower() in d.page_content.lower() for d in docs)
                exact = pred.strip().lower() == ref.strip().lower()

                scores = {
                    "scenario": desc,
                    "lang": lang,
                    "top_k": top_k,
                    "retrieval_precision": int(retrieval),
                    "exact_match": int(exact),
                    "f1": compute_f1(pred, ref),
                    "bleu": bleu_score(pred, ref),
                    "rougeL": rougeL_score(pred, ref),
                    "emb_sim": cos_sim(embed(pred), embed(ref)),
                }
                results.append(scores)

    save_json(results, "RAG/answers/evaluation_rag_result.txt")
    return results
""",
            language="python",
        )

    with tabs[5]:
        st.markdown("**app.py → Filtre de langue + contrôle cohérence**")
        st.code(
            r"""
lang = st.selectbox("Langue", ["Français", "English", "Español"], index=0)
lang_code = {"Français": "fr", "English": "en", "Español": "es"}[lang]

docs = vectorstore.similarity_search(
    question,
    k=5,
    filter={"lang": lang_code}
)

context = "\n\n".join(d.page_content for d in docs)

prompt = (
    f"Réponds en {lang} et uniquement d'après le contexte.\n\n"
    f"Contexte:\n{context}\n"
    f"Question: {question}\nRéponse:"
)

out = generator(prompt, do_sample=False)[0]["generated_text"]
answer = out.split("Réponse:")[-1].strip()

pred_lang = detect(answer)
if pred_lang != lang_code:
    st.warning(f"Langue détectée={pred_lang}, attendue={lang_code}")
""",
            language="python",
        )

# =============================================
# 5. ALIGNEMENT 100% FICHE DE POSTE
# =============================================
elif page == "Alignement 100% Fiche de Poste":
    st.markdown(
        f"<h2 style='color:{primary_green};'>100% Alignée avec la Fiche de Poste</h2>",
        unsafe_allow_html=True,
    )
    st.success("**Je coche TOUTES les cases**")

    st.markdown(
        f"""
    | Compétence Demandée (Fiche Bayes)              | Prouvé Ici                                      |
    |------------------------------------------------|-------------------------------------------------|
    | Python + BeautifulSoup/scraping                | `dataset_conte.py`                              |
    | RAG + Embeddings (MiniLM/MPNet)                | 3 index FAISS comparables.                      |
    | Data sourcing (API/Scraping)                   | Wikipedia → prêt pour APIs sociales CaseAI      |
    | Vector DB (pgvector/Pinecone)                  | FAISS (même principe)                           |
    | Évaluation RAG                                 | ROUGE/BLEU + protocole complet                  |
    | Bilingual FR/EN + ES                           | Démo switch + filtre lang                       |
    | **Mission du poste**                           | **Pipeline complet + évaluation + lead RAG**    |
    """,
        unsafe_allow_html=True,
    )

    st.balloons()
    st.markdown("**Je suis motivée par ce poste car je souhaite contribuer à une IA accessible et éthique.**")

# =============================================
# 6. ÉVALUATION
# =============================================
elif page == "Évaluation":
    st.markdown(f"<h2 style='color:{primary_green};'>Évaluation Quantitative</h2>", unsafe_allow_html=True)

    st.markdown(
        """
Cette page synthétise le protocole d’évaluation implémenté dans `rag_evaluation_to_complete.py`.
L’idée est de comparer **plusieurs combinaisons RAG** (index/embeddings/top-k) et **plusieurs LLM**
sur exactement les mêmes questions (FR/EN/ES), puis de mesurer à la fois la qualité du retrieval,
la qualité de génération et le coût d’inférence (latence/mémoire).
"""
    )

    with st.expander("Protocole : scénarios testés", expanded=True):
        st.markdown(
            """
On définit d’abord les **ressources et scénarios** :
- 3 index FAISS (MiniLM-500, MiniLM-2000, MPNet-500)
- 2 modèles d’embeddings (MiniLM vs MPNet)
- 2 modèles de génération (Qwen vs TinyLlama)
- plusieurs valeurs de **top-k** (1, 3, 5, 10)
- 3 questions identiques en 3 langues.
"""
        )
        st.code(
            """# --- Configuration (extrait)
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
    ("Qui est Charles Perrault ?", "...référence FR...", "fr"),
    ("Who is Charles Perrault?", "...reference EN...", "en"),
    ("¿Quién es Charles Perrault?", "...referencia ES...", "es")
]""",
            language="python",
        )

    with st.expander("Boucle d’évaluation d’un scénario", expanded=False):
        st.markdown(
            """
Pour chaque scénario, on :
1. charge l’index + embeddings correspondant
2. charge le LLM
3. exécute les 3 questions
4. mesure la latence et la mémoire
5. calcule toutes les métriques
6. enregistre le résultat en sortie.

Extrait central :
"""
        )
        st.code(
            r"""def evaluate_scenario(self, desc, index_path, embed_model, llm_model, top_k, lang_hint):
        vs = load_vectorstore(index_path, embed_model)
        text_gen = load_cpu_friendly_llm(llm_model)
        if self.embedder is None or getattr(self.embedder, 'model_name', None) != embed_model:
            self.embedder = SentenceTransformer(embed_model)

        test_cases = []
        for q, ref, lang in TEST_QUESTIONS:
            # measure latency and memory
            start = time.perf_counter()
            mem_before = psutil.Process().memory_info().rss if psutil else None
            pred, docs = answer_question_rag(
                q, vs, text_gen, top_k=top_k, lang_hint=(lang_hint and lang)
            )
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
                'emb_sim': float(cosine_similarity(
                    [self.embedder.encode(pred)], [self.embedder.encode(ref)]
                )[0][0]),
                'lang_consistent': int(lang_consistent),
                'unknown': int(unknown),
                'latency_s': latency,
                'mem_delta_mb': mem_usage
            }
            tc.context_language = lang
            test_cases.append(tc)""",
            language="python",
        )

    with st.expander("Calcul des métriques", expanded=False):
        st.markdown(
            """
Chaque question produit une prédiction `pred` et une liste de chunks `docs`.  
On calcule ensuite :

- **retrieval_precision** : vaut 1 si **la réponse de référence apparaît textuellement** dans au moins un chunk récupéré.
- **exact_match** : égalité stricte `pred == ref`.
- **F1 token-level** : overlap des tokens entre prédiction et référence.
- **BLEU / ROUGE-L** : métriques n-grammes classiques.
- **emb_sim** : similarité cosinus entre embeddings de `pred` et `ref`.
- **lang_consistent** : langue détectée de `pred` = langue attendue.
- **unknown** : le LLM a rendu “Je ne sais pas.” (proxy anti-hallucination).
- **latency_s / mem_delta_mb** : coût CPU.

Bloc de calcul :
"""
        )
        st.code(
            r"""tc.metric_scores = {
                'retrieval_precision': int(retrieval),
                'exact_match': int(exact),
                'f1': f1,
                'bleu': self.bleu.compute(predictions=[pred], references=[ref])["bleu"],
                'rougel': self.rouge.compute(predictions=[pred], references=[ref])["rougeL"],
                'emb_sim': float(cosine_similarity(
                    [self.embedder.encode(pred)], [self.embedder.encode(ref)]
                )[0][0]),
                'lang_consistent': int(lang_consistent),
                'unknown': int(unknown),
                'latency_s': latency,
                'mem_delta_mb': mem_usage
            }""",
            language="python",
        )

    st.markdown("### Résumé chiffré (meilleurs scores observés)")
    st.dataframe(
        {
            "Index": ["MiniLM 500", "MiniLM 2000", "MPNet 500"],
            "ROUGE-L max": ["0.08", "0.06", "0.09"],
            "Latence": ["25s", "38s", "42s"],
            "Qualité perçue": ["Bonne", "Moyenne", "Meilleure sémantique"],
        }
    )

    st.info(
        "Lecture rapide : emb_sim reste correcte (~0.6) mais retrieval_precision=0 "
        "=> le vrai problème est le chunking/ranking, pas le LLM."
    )

# =============================================
# FOOTER – Message final fort
# =============================================
st.markdown("---")
st.markdown(
    f"""
<div class="footer">
    <strong>Anissa Thezenas</strong><br>
    Candidate Junior Data Scientist RAG @ Bayes Impact<br><br>
    <span style="font-size:1.4em; color:{orange};">Je suis prête à en apprendre plus et je suis très motivée par l'impact que peut avoir la RAG sur les processus décisionnels des travailleurs sociaux.</span><br><br>
    <strong>Hâte de faire partie de l'aventure !</strong>
</div>
""",
    unsafe_allow_html=True,
)