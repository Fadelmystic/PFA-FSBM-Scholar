# Diagrammes Mermaid - Pipelines FSBM Scholar Assistant

## 1. Pipeline Graphes de Connaissances (Knowledge Graph)

```mermaid
flowchart TD
    A[Documents FSBM<br/>PDF, DOCX, PPTX] --> B[Extraction de texte<br/>PyPDF2, python-docx]
    B --> C[Prétraitement<br/>Nettoyage, normalisation]
    C --> D[Analyse linguistique<br/>spaCy, NLTK]
    
    D --> E[Extraction d'entités<br/>NER, concepts académiques]
    D --> F[Détection de relations<br/>Dépendances, prérequis]
    
    E --> G[Construction des nœuds<br/>Modules, concepts, enseignants]
    F --> H[Construction des arêtes<br/>Précède, nécessite, enseigne]
    
    G --> I[Base de données graphe<br/>Neo4j]
    H --> I
    
    I --> J[Validation et cohérence<br/>Vérification des relations]
    J --> K[Enrichissement ontologique<br/>Métadonnées, propriétés]
    
    K --> L[Graphe de connaissances FSBM<br/>Entités connectées]
    
    style A fill:#e1f5fe
    style L fill:#c8e6c9
    style I fill:#fff3e0
```

## 2. Pipeline RAG (Retrieval Augmented Generation)

```mermaid
flowchart TD
    A[Question utilisateur<br/>Interface de chat] --> B[Prétraitement<br/>Nettoyage, normalisation]
    
    B --> C[Vectorisation de la question<br/>multilingual-e5-small]
    
    C --> D[Recherche vectorielle<br/>Base FAISS]
    D --> E[Récupération des chunks<br/>Top-K documents pertinents]
    
    E --> F[Reranking intelligent<br/>Score de similarité cosinus]
    F --> G[Sélection des sources<br/>Seuil de pertinence > 0.35]
    
    G --> H[Construction du prompt<br/>Question + Sources + Instructions]
    
    H --> I[Génération LLaMA 3.1<br/>OpenRouter API]
    I --> J[Post-traitement<br/>Formatage, citations]
    
    J --> K[Réponse finale<br/>Sources citées, traçabilité]
    
    L[Base vectorielle FAISS<br/>500+ documents indexés] --> D
    M[Modèle d'embedding<br/>intfloat/multilingual-e5-small] --> C
    
    style A fill:#e1f5fe
    style K fill:#c8e6c9
    style L fill:#fff3e0
    style M fill:#fff3e0
```

## 3. Pipeline Fine-tuning LLaMA avec LoRA

```mermaid
flowchart TD
    A[Dataset FSBM<br/>faduil/fsbm-qa-dataset] --> B[Préparation des données<br/>Format LLaMA chat]
    
    B --> C[Tokenisation<br/>Tokenizer LLaMA 2-7B]
    C --> D[Chunking intelligent<br/>256 tokens max, padding]
    
    D --> E[Split train/validation<br/>90% train, 10% val]
    
    E --> F[Chargement modèle base<br/>LLaMA 2-7B-chat-hf]
    F --> G[Configuration 8-bit<br/>BitsAndBytesConfig]
    
    G --> H[Configuration LoRA<br/>r=16, alpha=32, dropout=0.1]
    H --> I[Application PEFT<br/>get_peft_model]
    
    I --> J[Configuration entraînement<br/>TrainingArguments]
    J --> K[Trainer PEFT<br/>8 époques, batch_size=1]
    
    K --> L[Entraînement LoRA<br/>Gradient accumulation 8x]
    L --> M[Sauvegarde adapters<br/>Modèle fine-tuné]
    
    M --> N[Test et validation<br/>Questions académiques FSBM]
    N --> O[Modèle adapté FSBM<br/>Vocabulaire académique]
    
    style A fill:#e1f5fe
    style O fill:#c8e6c9
    style H fill:#fff3e0
    style M fill:#fff3e0
```

## 4. Architecture Globale du Système

```mermaid
graph TB
    subgraph "Interface Utilisateur"
        A[Interface Web<br/>Streamlit/FastAPI]
        B[Chatbot conversationnel<br/>Français/Anglais]
    end
    
    subgraph "Backend API"
        C[FastAPI Server<br/>Gestion des requêtes]
        D[Gestion des sessions<br/>Historique utilisateur]
    end
    
    subgraph "Pipeline KG-RAG-LLM"
        E[Question Processing<br/>Nettoyage, normalisation]
        F[Knowledge Graph Query<br/>Neo4j, relations]
        G[RAG Retrieval<br/>FAISS, embeddings]
        H[Fusion des résultats<br/>KG + RAG]
        I[LLM Generation<br/>LLaMA 3.1 fine-tuné]
    end
    
    subgraph "Bases de Données"
        J[Base vectorielle FAISS<br/>500+ documents]
        K[Graphe de connaissances<br/>Neo4j]
        L[Cache des conversations<br/>SQLite/JSON]
    end
    
    subgraph "Modèles et Services"
        M[Modèle d'embedding<br/>multilingual-e5-small]
        N[LLaMA 3.1 via OpenRouter<br/>API externe]
        O[Modèle fine-tuné LoRA<br/>Adaptation FSBM]
    end
    
    subgraph "Pipeline de Données"
        P[Extraction documents<br/>PDF, DOCX, PPTX]
        Q[Prétraitement<br/>Chunking, nettoyage]
        R[Vectorisation<br/>Embeddings, indexation]
        S[Construction KG<br/>Entités, relations]
    end
    
    %% Flux de données
    A --> C
    B --> C
    C --> E
    E --> F
    E --> G
    F --> H
    G --> H
    H --> I
    
    F --> K
    G --> J
    I --> N
    I --> O
    
    M --> G
    M --> R
    
    P --> Q
    Q --> R
    R --> J
    Q --> S
    S --> K
    
    %% Retour utilisateur
    I --> C
    C --> A
    C --> B
    
    %% Styles
    classDef interface fill:#e1f5fe
    classDef backend fill:#fff3e0
    classDef pipeline fill:#f3e5f5
    classDef database fill:#e8f5e8
    classDef model fill:#fff8e1
    classDef data fill:#fce4ec
    
    class A,B interface
    class C,D backend
    class E,F,G,H,I pipeline
    class J,K,L database
    class M,N,O model
    class P,Q,R,S data
```

## 5. Flux de Données Détaillé

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant I as Interface Web
    participant A as API FastAPI
    participant KG as Knowledge Graph
    participant R as RAG Pipeline
    participant L as LLaMA 3.1
    participant F as Base FAISS
    participant N as Neo4j
    
    U->>I: Pose une question
    I->>A: Envoie la requête
    A->>A: Prétraitement (nettoyage)
    
    par Recherche parallèle
        A->>KG: Interroge le graphe
        KG->>N: Requête Cypher
        N-->>KG: Relations et entités
        KG-->>A: Résultats structurés
    and
        A->>R: Lance la recherche RAG
        R->>F: Recherche vectorielle
        F-->>R: Documents pertinents
        R->>R: Reranking et sélection
        R-->>A: Sources documentaires
    end
    
    A->>A: Fusion KG + RAG
    A->>L: Génération avec contexte
    L-->>A: Réponse générée
    A->>A: Post-traitement et formatage
    A-->>I: Réponse avec sources
    I-->>U: Affichage de la réponse
```

---

## Description des Composants

### **Pipeline KG :**
- **Extraction** : Traitement des documents FSBM
- **Construction** : Création des nœuds et relations
- **Validation** : Vérification de la cohérence
- **Enrichissement** : Ajout de métadonnées

### **Pipeline RAG :**
- **Recherche** : Vectorisation et indexation FAISS
- **Récupération** : Sélection des documents pertinents
- **Génération** : LLaMA 3.1 avec contexte
- **Traçabilité** : Citations des sources

### **Pipeline Fine-tuning :**
- **Préparation** : Dataset académique FSBM
- **Configuration** : LoRA avec PEFT
- **Entraînement** : Adaptation ciblée
- **Validation** : Test sur questions académiques

### **Architecture Globale :**
- **Interface** : Chatbot conversationnel
- **Backend** : API FastAPI orchestratrice
- **Pipelines** : KG, RAG, LLM intégrés
- **Bases** : FAISS, Neo4j, cache
- **Modèles** : Embeddings, LLaMA fine-tuné

Ces diagrammes illustrent clairement la complexité et l'innovation de votre approche hybride KG-RAG-LLM ! 🚀

## 6. Diagramme de Classes (UML)

```mermaid
classDiagram
    direction LR

    class ChatController {
        +postChat(request: ChatRequest): ChatResponse
        +switchMode(mode: string): void
    }

    class RagService {
        -vectorStore: FaissIndex
        -embeddingModel: EmbeddingModel
        -llmClient: LlmClient
        +answer(query: Query): Response
        +retrieveChunks(query: Query): DocumentChunk[]
    }

    class KgService {
        -neo4j: Neo4jManager
        +answer(query: Query): Response
        +detectQueryType(text: string): string
        +extractConcepts(text: string): string[]
    }

    class FineTuningService {
        -dataset: Dataset
        -trainer: Trainer
        +train(config: TrainingArgs): ModelAdapter
        +loadAdapter(path: string): ModelAdapter
    }

    class FaissIndex {
        +add(chunks: DocumentChunk[]): void
        +search(vector: number[], topK: int, threshold: float): DocumentChunk[]
    }

    class EmbeddingModel {
        +embed(text: string): number[]
        +embedBatch(texts: string[]): number[][]
    }

    class LlmClient {
        +generate(prompt: string): string
        +modelName: string
    }

    class Neo4jManager {
        +query(cypher: string, params: map): Result
        +upsertNode(labels: string[], props: map): void
        +upsertRelation(a: NodeRef, b: NodeRef, type: string, props: map): void
    }

    class DocumentIngestionPipeline {
        +extract(path: string): string
        +clean(text: string): string
        +chunk(text: string): DocumentChunk[]
        +index(chunks: DocumentChunk[]): void
        +updateKg(chunks: DocumentChunk[]): void
    }

    class Config {
        +openrouterApiKey: string
        +neo4jUri: string
        +faissPath: string
        +similarityThreshold: float
        +topK: int
    }

    class Logger {
        +info(msg: string): void
        +error(msg: string): void
    }

    class Query {
        +text: string
        +mode: string
        +filters: map
    }

    class Response {
        +text: string
        +sources: string[]
        +mode: string
    }

    class DocumentChunk {
        +id: string
        +content: string
        +embedding: number[]
        +metadata: map
    }

    class Concept {
        +name: string
        +definition: string
    }

    class Relation {
        +type: string
        +description: string
    }

    %% Relations
    ChatController --> RagService : utilise
    ChatController --> KgService : utilise
    ChatController --> FineTuningService : utilise

    RagService --> FaissIndex : index/recherche
    RagService --> EmbeddingModel : embeddings
    RagService --> LlmClient : génération

    KgService --> Neo4jManager : requêtes Cypher

    FineTuningService --> LlmClient : modèle base
    FineTuningService --> EmbeddingModel : tokenisation/embeddings

    DocumentIngestionPipeline --> FaissIndex : indexation
    DocumentIngestionPipeline --> Neo4jManager : construction KG

    FaissIndex --> DocumentChunk : stocke
    Neo4jManager --> Concept : nœuds
    Neo4jManager --> Relation : arêtes

    ChatController ..> Config
    RagService ..> Config
    KgService ..> Config
    FineTuningService ..> Config
    ChatController ..> Logger
```


