# RAG et évaluation du RAG sur le corpus de Charles Perrault

Ce projet met en œuvre un pipeline **Retrieval-Augmented Generation (RAG)** multilingue visant à répondre à des questions factuelles sur Charles Perrault et ses contes emblématiques. Il s’appuie sur un index FAISS pour le retrieval de passages et un modèle de génération (LLM) pour produire des réponses contextualisées en français, anglais et espagnol.

---

## Table des matières

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Structure du dépôt](#structure-du-dépôt)
4. [Préparation du corpus](#préparation-du-corpus)
5. [Indexation FAISS](#indexation-faiss)
6. [Génération de réponses](#génération-de-réponses)
7. [Protocole d’évaluation](#protocole-d%C3%A9valuation)
8. [Exécution complète](#ex%C3%A9cution-compl%C3%A8te)
9. [Résultats](#r%C3%A9sultats)

---

## Prérequis

* Python 3.8 ou supérieur
* 8 Go de RAM (CPU uniquement)
* Accès Internet pour télécharger les modèles Transformers

Installez les dépendances :

```bash
pip install \
  langchain-community \
  transformers \
  sentence-transformers \
  faiss-cpu \
  langdetect \
  evaluate \
  torch
```


## Préparation du corpus

Le dossier `data/` doit contenir :

* **fr/** : textes originaux en français (biographie + 4 contes)
* **en/** : traductions de référence en anglais
* **es/** : traductions de référence en espagnol

Les fichiers `.txt` sont chargés récursivement puis segmentés en fragments (chunks) de 500 ou 2000 caractères (chevauchement 200).

## Indexation FAISS

L’index FAISS se construit et se charge via :

```bash
# Construction de l’index
python src/rag.py --build-index \
  --data-path data/ \
  --index-path faiss_index/ \
  --chunk-size 500 --chunk-overlap 200 \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Chargement de l’index existant
python src/rag.py \
  --index-path faiss_index/ \
  --question "Qui est Charles Perrault ?"
```

Plusieurs index sont disponibles : `faiss_index/` (MiniLM), `faiss_index_mpnet/` (MPNet).

## Génération de réponses

Pour interroger le système RAG :

```bash
python src/rag.py \
  --index-path faiss_index/ \
  --llm-model Qwen/Qwen2.5-1.5B-Instruct \
  --question "Who is Charles Perrault?" \
  --lang en --top-k 3
```

Les réponses s’affichent et sont enregistrées dans `answers/`.

## Protocole d’évaluation

Le protocole d’évaluation mis en place pour ce projet RAG comporte trois volets : les scénarios de test, les questions de test et les métriques de performance. L’objectif est de mesurer rigoureusement l’impact des paramètres de retrieval, du choix des embeddings et du modèle de génération sur la qualité des réponses fournies.

### 1. Scénarios de Test
Quatre scénarios ont été explorés :

- **Variation du paramètre top-k**  
  Évaluer l’influence du nombre de fragments récupérés sur la pertinence des réponses.  
  Valeurs testées : top-k ∈ {1, 3, 5, 10}.

- **Comparaison des embeddings MiniLM vs MPNet**  
  Comparer deux modèles d’embedding en maintenant constant le reste de la configuration (LLM, taille des chunks, top-k).

- **Comparaison des LLM Qwen vs TinyLlama**  
  Mesurer latence, empreinte mémoire et qualité de génération pour chaque modèle ; embeddings, chunking et top-k étant identiques.

- **Désactivation du filtre de langue**  
  Vérifier si le système reste cohérent lorsque l’on force ou non la réponse dans la langue de la question (paramètre lang_hint).

### 2. Questions de Test
Pour chaque scénario, trois questions factuelles ont été posées, formulées dans les trois langues cibles :

- **Français** : « Qui est Charles Perrault ? »  
- **Anglais** : “Who is Charles Perrault?”  
- **Espagnol** : “¿Quién es Charles Perrault?”  

Ces questions simples permettent de juger à la fois du retrieval d’informations précises et de la capacité de génération multilingue.

### 3. Métriques de Performance
Pour chaque exécution, les indicateurs suivants ont été mesurés :

| **Métrique**         | **Description**                                                                 |
|----------------------|---------------------------------------------------------------------------------|
| **retrieval_precision** | Proportion de fragments récupérés jugés pertinents par rapport à la réponse attendue. |
| **exact_match**      | 1 si la réponse générée correspond exactement à la référence, 0 sinon.          |
| **F1 (token-level)** | Moyenne harmonique de la précision et du rappel au niveau des tokens entre réponse générée et référence. |
| **BLEU**             | Score de similarité n-grammes entre la réponse générée et la référence.         |
| **ROUGE-L**          | Longueur de la plus longue sous-séquence commune (LCS) entre réponse générée et référence. |
| **emb_sim**          | Similarité cosinus entre embeddings de la réponse générée et de la référence (valeur comprise entre 0 et 1). |
| **lang_consistent**  | 1 si la langue de la réponse générée correspond à la langue de la question, 0 sinon. |
| **unknown**          | Taux de réponses « Je ne sais pas » ; idéalement, élevé en cas d’absence totale d’information, mais ici toujours nul. |
| **latency_s**        | Temps de génération de la réponse (en secondes).                                |
| **mem_delta_mb**     | Variation de la consommation de mémoire vive (en Mo) entre le début et la fin de la génération. |


## Exécution complète

Pour lancer la chaîne complète (indexation + génération + évaluation) :

```bash
# Construire les index
python src/rag.py --build-index --data-path data/ --index-path faiss_index_*/ ...
python src/rag.py --build-index --data-path data/ --index-path faiss_index_mpnet/ --embedding-model paraphrase-multilingual-mpnet-base-v2 ...

# Exécuter l’évaluation
python src/rag_evaluation_to_complete.py
```

## Résultats

Les résultats quantitatifs (scores et latences) sont enregistrés dans `answers/evaluation_rag_result.txt`.
