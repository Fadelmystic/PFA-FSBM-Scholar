### Annexe — Guide d’installation et d’utilisation

Ce guide explique comment installer et exécuter les composants de l’assistant FSBM : API RAG (FastAPI), API Knowledge Graph (Flask + Neo4j) et interface Streamlit.

---

## Prérequis
- **Python 3.10+** (recommandé 3.10/3.11)
- **pip** et **virtualenv** (ou venv)
- **Git**
- **FAISS** (installé via pip; binaire présent dans `vectorstore/db_faiss`)
- (Optionnel) **Neo4j** pour le Knowledge Graph
- (Optionnel) **Docker Desktop** si vous souhaitez lancer Neo4j en conteneur

---

## Arborescence pertinente
- `chatbot/api.py` — API RAG (FastAPI, port par défaut 8000)
- `chatbot/config.py` — variables d’environnement (OpenRouter, chemins, etc.)
- `chatbot/connect_memory_with_llm_lighter_cos.py` — RAG (FAISS, embeddings)
- `chatbot/knowledge_graph/kg_api.py` — API KG (Flask, port par défaut 8001)
- `chatbot/knowledge_graph/config.py` — config Neo4j
- `vectorstore/db_faiss` — index FAISS pré‑généré

---

## Installation

1) Cloner le projet
```bash
git clone <votre_repo_ou_chemin_local>
cd fsbm-scholar-assistant
```

2) Créer et activer un environnement virtuel
- Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Installer les dépendances
- Dépendances générales
```bash
pip install -r chatbot/requirements.txt
```
- Dépendances Knowledge Graph (si utilisation KG)
```bash
pip install -r chatbot/knowledge_graph/requirements.txt
```
- Dépendances pipeline KG (extraction automatique — optionnel)
```bash
pip install -r chatbot/knowledge_graph/pipeline_requirements.txt
```
- Dépendances Fine‑tuning (si vous utilisez la partie `Fine-tuning/` — optionnel)
```bash
pip install -r chatbot/Fine-tuning/requirements.txt
```

---

## Configuration de l’environnement
Créez un fichier `.env` à la racine du dossier `chatbot/` avec vos clés et options.

Exemple minimal pour le RAG et le KG :
```dotenv
# RAG / OpenRouter
OPENROUTER_API_KEY=remplacez_par_votre_cle
OPENROUTER_MODEL=meta-llama/llama-3.1-405b-instruct:free

# API
API_HOST=0.0.0.0
API_PORT=8000

# Vector store
VECTORSTORE_PATH=vectorstore/db_faiss

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Knowledge Graph (Neo4j)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme
```

Notes :
- Une clé OpenRouter valide est requise pour les réponses LLM en production.
- Le chemin `VECTORSTORE_PATH` doit pointer vers `vectorstore/db_faiss` (déjà fourni).

---

## Lancer les services

### 1) API RAG (FastAPI)
Depuis le dossier `chatbot/` :
```powershell
# Windows PowerShell
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Ou (avec les valeurs par défaut dans `api.py`) :
```powershell
python chatbot\api.py
```
Accès : `http://localhost:8000`
- Endpoint test : `POST /chat`
- Endpoint santé : `GET /api/accueil`

### 2) API Knowledge Graph (Flask + Neo4j)
Prérequis : Neo4j en exécution.

- Option A — Neo4j local
  - Installer Neo4j Desktop/Server, créer une base, et définir `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` dans `.env`.

- Option B — Docker (exemple rapide)
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme neo4j:5
```

Lancer l’API KG depuis `chatbot/knowledge_graph/` :
```powershell
python kg_api.py
```
Accès : `http://localhost:8001`
- Santé : `GET /health`
- Chat : `POST /chat`
- Concepts : `GET /kg/concepts`

### 3) Interface Streamlit (RAG UI)
Depuis `chatbot/` :
```powershell
streamlit run connect_memory_with_llm_lighter_cos.py
```
Accès : `http://localhost:8501`

---

## Vérifications rapides

- API RAG — accueil
```bash
curl http://localhost:8000/api/accueil
```
- API RAG — chat
```bash
curl -X POST http://localhost:8000/chat \ 
  -H "Content-Type: application/json" \ 
  -d '{"message":"Qu 19est-ce que Hadoop?","similarity_threshold":0.35,"max_docs":5}'
```
- API KG — santé
```bash
curl http://localhost:8001/health
```
- API KG — chat
```bash
curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -d '{"message":"graphes de connaissances"}'
```

---

## Dépannage
- **OPENROUTER_API_KEY manquante** : l’API RAG retournera 500. Ajoutez la clé dans `chatbot/.env`.
- **Index FAISS introuvable** : vérifiez `VECTORSTORE_PATH` et l’existence de `vectorstore/db_faiss`.
- **Neo4j non connecté** : `GET /health` indique `neo4j_connected: false`. Vérifiez `NEO4J_URI/USER/PASSWORD` et que le service est en ligne.
- **Port déjà utilisé** : changez le port (`--port 8002` par exemple) ou libérez‐le.
- **Modules manquants** : réinstallez les requirements appropriés.

---

## Maintenance
- Mettre à jour le code
```bash
git pull
```
- Mettre à jour les dépendances
```bash
pip install -r chatbot/requirements.txt --upgrade
```
- Sauvegarde FAISS (copie du dossier)
```bash
cp -r vectorstore/db_faiss backups/db_faiss_$(date +%Y%m%d)
```

---

## Commandes utiles (Windows PowerShell)
- Activer venv : `\.venv\Scripts\Activate.ps1`
- Désactiver venv : `deactivate`
- Lancer FastAPI : `python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload`
- Lancer KG : `python chatbot\knowledge_graph\kg_api.py`
- Lancer Streamlit : `streamlit run chatbot\connect_memory_with_llm_lighter_cos.py`

---

## Démarrage du Frontend (standalone‑chat)
Le frontend minimal est une page statique qui appelle les APIs RAG et KG.

Emplacement :
- `standalone-chat/index.html`
- `standalone-chat/app.js`
- `standalone-chat/styles.css`

Pré‑requis :
- Démarrer l’API RAG sur `http://localhost:8000` (section ci‑dessus)
- Démarrer l’API KG sur `http://localhost:8001` (section ci‑dessus)

Options de démarrage :

- Option A — Ouvrir le fichier directement
  - Double‑cliquez `standalone-chat/index.html` et utilisez votre navigateur.
  - Si votre navigateur bloque les requêtes locales, préférez l’Option B.

- Option B — Servir avec un petit serveur HTTP (recommandé)
  - Depuis la racine du projet :
```powershell
# Windows PowerShell
cd standalone-chat
python -m http.server 5500
```
  - Ouvrir `http://localhost:5500` dans le navigateur.

Dans l’interface :
- Bouton « RAG » : envoie vers `http://localhost:8000/chat`
- Bouton « KG » : envoie vers `http://localhost:8001/chat`

Astuce : si vous changez les ports des APIs, ajustez les constantes `RAG_URL` et `KG_URL` dans `standalone-chat/app.js`.

---

## APIs de test — Récapitulatif rapide

### API RAG (FastAPI — `chatbot/api.py`)
Base : `http://localhost:8000`
- `GET /api/accueil`
```bash
curl http://localhost:8000/api/accueil
```
- `POST /chat`
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explique Hadoop en 2 lignes","similarity_threshold":0.35,"max_docs":5}'
```

### API Knowledge Graph (Flask — `chatbot/knowledge_graph/kg_api.py`)
Base : `http://localhost:8001`
- `GET /health`
```bash
curl http://localhost:8001/health
```
- `POST /chat`
```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"qu\u00e9saco graphes de connaissances"}'
```
- `POST /kg/query`
```bash
curl -X POST http://localhost:8001/kg/query \
  -H "Content-Type: application/json" \
  -d '{"query":"NoSQL vs SQL"}'
```
- `GET /kg/concepts`
```bash
curl http://localhost:8001/kg/concepts
```
- `POST /kg/search`
```bash
curl -X POST http://localhost:8001/kg/search \
  -H "Content-Type: application/json" \
  -d '{"query":"hadoop"}'
```

### UI Streamlit (facultatif — `chatbot/connect_memory_with_llm_lighter_cos.py`)
- Lancer :
```powershell
cd chatbot
streamlit run connect_memory_with_llm_lighter_cos.py
```
- Accès : `http://localhost:8501`

---

## Vérification du flux complet
1) Démarrer RAG (`:8000`) et KG (`:8001`).
2) Servir `standalone-chat/` (ex : `http://localhost:5500`).
3) Dans le frontend :
   - Sélectionner « RAG » et poser une question cours → réponse RAG.
   - Sélectionner « KG » et poser une question concept → réponse KG.
