# CHAPITRE III : ÉTUDE PRÉALABLE DU PROJET

## 3.1 INTRODUCTION

Dans ce chapitre, nous abordons les principes fondamentaux de notre sujet en explorant son contexte et ses bases. Notre objectif est de saisir les concepts clés qui constituent la base de notre travail, tout en établissant clairement le cadre dans lequel notre projet FSBM Scholar Assistant s'inscrit.

Cette étude préalable nous permet de comprendre les technologies sous-jacentes, de justifier nos choix architecturaux et de présenter les fondements conceptuels de l'approche hybride KG-RAG-LLM que nous proposons. Nous explorerons successivement les Large Language Models, les Graphes de Connaissances, l'architecture RAG, et les spécificités de notre implémentation.

## 3.2 GÉNÉRALITÉS SUR LES LLM (LARGE LANGUAGE MODELS)

### 3.2.1 Définition et concepts fondamentaux

Les LLM (Large Language Models) sont des modèles avancés d'intelligence artificielle appartenant au domaine du traitement du langage naturel (NLP). Ces modèles sont capables de comprendre et de générer du texte en se basant sur des modèles statistiques obtenus par l'entraînement sur d'immenses corpus textuels.

**Caractéristiques principales :**
- **Modèles de base** : Entraînés sur des corpus massifs de textes
- **Architecture Transformer** : Basée sur des mécanismes d'attention
- **Capacités émergentes** : Raisonnement, compréhension contextuelle, génération créative
- **Transfer learning** : Adaptation à des tâches spécifiques via fine-tuning

### 3.2.2 Principes de base et fonctionnement

**Architecture Transformer :**
```
[Input Tokens] → [Embedding Layer] → [Multi-Head Attention] → [Feed Forward] → [Output]
                      ↓                      ↓                    ↓
                  Vectorisation         Mécanisme d'attention   Traitement
                  des tokens           (Query, Key, Value)     des relations
```

**Mécanismes clés :**
- **Self-Attention** : Modélisation des relations entre tokens
- **Positional Encoding** : Préservation de l'ordre séquentiel
- **Multi-Head Attention** : Parallélisation des calculs d'attention
- **Layer Normalization** : Stabilisation de l'entraînement

**Processus de génération :**
1. **Tokenisation** : Conversion du texte en tokens numériques
2. **Embedding** : Représentation vectorielle des tokens
3. **Attention** : Calcul des relations contextuelles
4. **Décodage** : Génération token par token avec sampling

### 3.2.3 Défis et limitations des LLM

Bien que puissants, les LLM comme LLaMA 3.1 présentent certaines limites et défis importants :

**Limitations contextuelles :**
- **Fenêtre contextuelle** : Limitation de la longueur des séquences traitées
- **Perte d'information** : Oubli des informations en début de séquence
- **Interprétation contextuelle** : Difficulté à maintenir la cohérence sur de longs textes

**Biais et éthique :**
- **Biais intrinsèques** : Reflet des biais présents dans les données d'entraînement
- **Hallucinations** : Génération d'informations non factuelles
- **Toxicité** : Production de contenu inapproprié ou dangereux

**Dépendances et contraintes :**
- **Qualité des données** : Dépendance à la qualité du corpus d'entraînement
- **Ressources computationnelles** : Coût élevé de l'entraînement et de l'inférence
- **Actualité** : Limitation à la date de fin des données d'entraînement

## 3.3 GÉNÉRALITÉS SUR LLaMA 3.1

### 3.3.1 Introduction et spécificités

Dans le cadre précis de notre projet, nous avons choisi d'utiliser spécifiquement LLaMA 3.1 via l'API d'OpenRouter. LLaMA 3.1 est une évolution majeure des modèles de la famille LLaMA de Meta, spécialement optimisée pour les applications de chatbot et la génération contextualisée de réponses fiables et pertinentes.

**Caractéristiques techniques LLaMA 3.1 :**
- **Architecture** : Transformer Decoder-Only optimisé
- **Paramètres** : Modèles de 8B à 70B paramètres
- **Fenêtre contextuelle** : 8K à 32K tokens selon la variante
- **Entraînement** : Corpus multilingue français/anglais
- **Optimisations** : Grouped Query Attention, RMSNorm

**Avantages pour notre projet :**
- **Performance équilibrée** : Bon rapport qualité/ressources
- **Support multilingue** : Français et anglais natifs
- **Sécurité renforcée** : Mécanismes anti-toxicité intégrés
- **Accessibilité** : Disponible via OpenRouter avec coûts maîtrisés

### 3.3.2 Comparaison avec d'autres modèles

**Critères de comparaison :**

| Critère | LLaMA 3.1 | GPT-3.5 | Claude 3 Haiku |
|---------|------------|---------|----------------|
| **Taille des données** | Modérée | Élevée | Élevée |
| **Sécurité des réponses** | Élevée | Moyenne | Très élevée |
| **Fraîcheur des données** | 2024 | 2021 | 2023 |
| **Accessibilité** | OpenRouter | OpenAI API | Anthropic API |
| **Coût** | Modéré | Élevé | Modéré |
| **Support français** | Natif | Bon | Bon |
| **Fine-tuning** | Supporté | Limité | Limité |

**Justification du choix LLaMA 3.1 :**
- **Performance équilibrée** : Bonnes capacités sans surcoût
- **Sécurité académique** : Réponses appropriées pour l'éducation
- **Accessibilité** : Intégration simple via OpenRouter
- **Flexibilité** : Possibilité de fine-tuning pour le domaine FSBM

### 3.3.3 Intégration via OpenRouter

**Avantages de l'API OpenRouter :**
- **Interface unifiée** : Accès à plusieurs modèles via une seule API
- **Gestion des coûts** : Facturation transparente et prévisible
- **Scalabilité** : Adaptation automatique aux besoins
- **Sécurité** : Filtres et contrôles intégrés

**Configuration spécifique :**
```python
# Configuration OpenRouter pour LLaMA 3.1
OPENROUTER_API_KEY = "sk-or-v1-..."
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"
API_BASE = "https://openrouter.ai/api/v1"
```

## 3.4 GRAPHES DE CONNAISSANCES (KNOWLEDGE GRAPHS)

### 3.4.1 Concepts et principes

Les Graphes de Connaissances représentent une approche structurée de la représentation des informations, où les entités sont connectées par des relations explicites. Dans notre contexte académique, cette représentation permet de modéliser les relations entre concepts, modules et ressources pédagogiques.

**Structure des graphes :**
- **Nœuds (Entités)** : Concepts, modules, enseignants, ressources
- **Arêtes (Relations)** : Précède, nécessite, enseigne, couvre
- **Propriétés** : Métadonnées, scores, dates
- **Ontologies** : Schémas conceptuels du domaine

**Exemple de graphe FSBM :**
```
[Bigdata] --précède--> [Machine Learning]
[Python] --nécessite--> [Programmation de base]
[Cloud Computing] --couvre--> [AWS, Azure, GCP]
```

### 3.4.2 Implémentation dans notre projet

**Technologies utilisées :**
- **Neo4j** : Base de données graphe native
- **Cypher** : Langage de requête pour graphes
- **Python** : Interface de programmation
- **NetworkX** : Manipulation et visualisation

**Processus de construction :**
1. **Extraction d'entités** : Identification des concepts clés
2. **Détection de relations** : Analyse des dépendances
3. **Validation** : Vérification de la cohérence
4. **Enrichissement** : Ajout de métadonnées

## 3.5 ARCHITECTURE RAG (RETRIEVAL AUGMENTED GENERATION)

### 3.5.1 Principes et composants

L'architecture RAG combine efficacement la récupération d'information et la génération de réponses, assurant des réponses factuelles et traçables. Notre projet utilise cette méthode hybride pour garantir la pertinence et la précision des réponses générées.

**Composants de l'architecture RAG :**
```
[Question] → [Retriever] → [Documents] → [Generator] → [Réponse]
                ↓              ↓            ↓
            Embeddings    Base Vectorielle  LLM
            + Indexing    + Reranking     + Prompting
```

**Avantages de l'approche RAG :**
- **Factualité** : Réponses basées sur sources vérifiables
- **Traçabilité** : Citations et références explicites
- **Actualité** : Mise à jour continue des connaissances
- **Spécialisation** : Adaptation au domaine académique FSBM

### 3.5.2 Implémentation technique

**Pipeline de traitement :**
1. **Prétraitement** : Extraction et nettoyage du texte
2. **Chunking** : Découpage en segments de 800 tokens
3. **Embedding** : Vectorisation avec multilingual-e5-small
4. **Indexation** : Stockage dans FAISS
5. **Recherche** : Récupération par similarité cosinus
6. **Génération** : Réponse par LLaMA 3.1

**Technologies utilisées :**
- **FAISS** : Base de données vectorielle haute performance
- **Sentence Transformers** : Modèle d'embedding multilingue
- **NumPy** : Calculs de similarité et manipulation d'arrays
- **LangChain** : Orchestration du pipeline RAG

## 3.6 CHATBOT ET INTERFACE UTILISATEUR

### 3.6.1 Définition et fonctionnalités

Un chatbot est une interface conversationnelle qui permet aux utilisateurs d'interagir avec des systèmes numériques via du texte ou de la voix, imitant ainsi une interaction humaine naturelle. Dans notre projet, nous exploitons précisément le potentiel conversationnel de LLaMA 3.1 en combinaison avec l'approche RAG pour permettre des réponses précises basées sur des connaissances extraites directement des documents de référence.

**Fonctionnalités principales :**
- **Interface de chat** : Conversation naturelle en français/anglais
- **Recherche contextuelle** : Compréhension des questions complexes
- **Citations de sources** : Traçabilité des informations
- **Historique des conversations** : Suivi des interactions
- **Suggestions de questions** : Aide à la formulation

### 3.6.2 Architecture de l'interface

**Composants de l'interface :**
- **Frontend** : Interface web responsive (HTML/CSS/JavaScript)
- **Backend** : API FastAPI avec gestion des sessions
- **Base de données** : Stockage des conversations et métadonnées
- **Cache** : Optimisation des performances

**Flux d'interaction :**
1. **Saisie utilisateur** : Question ou demande
2. **Prétraitement** : Nettoyage et normalisation
3. **Recherche RAG** : Récupération des documents pertinents
4. **Génération** : Réponse par LLaMA 3.1
5. **Post-traitement** : Formatage et ajout des sources
6. **Affichage** : Présentation à l'utilisateur

## 3.7 DATASET ET PRÉPARATION DES DONNÉES

### 3.7.1 Composition du dataset

Dans notre projet, le terme Dataset désigne spécifiquement l'ensemble des documents (PDF, DOCX, PPTX, TXT) issus des supports pédagogiques de la FSBM, utilisés pour créer notre base de connaissances et notre graphe de connaissances.

**Types de documents inclus :**
- **PDF** : Cours, exercices, corrigés, présentations
- **PowerPoint** : Présentations de cours (.pptx)
- **Word** : Notes et documents de cours (.docx)
- **Fichiers techniques** : Code source, schémas (.vi, .m)

**Répartition par filière :**
- **BIGDATA** : ~110 documents (S1 + S2)
- **MNP** : ~400 documents (S1 + S2)
- **SMI** : ~18 documents (S6)
- **Total** : Plus de 500 documents académiques

### 3.7.2 Pipeline de préparation des données

**Étapes de traitement :**

1. **Extraction automatique du texte :**
   - **PDF** : PyPDF2, pdfplumber
   - **DOCX** : python-docx
   - **PPTX** : python-pptx
   - **TXT** : Lecture directe

2. **Prétraitement et nettoyage :**
   - Suppression des caractères spéciaux
   - Normalisation des espaces et retours à la ligne
   - Détection et préservation des formules mathématiques
   - Identification du langage (français/anglais)

3. **Chunking intelligent :**
   - **Taille des chunks** : 800 tokens par segment
   - **Chevauchement** : 80 tokens entre segments
   - **Préservation du contexte** : Respect des paragraphes
   - **Métadonnées** : Source, position, type de contenu

4. **Création d'embeddings :**
   - **Modèle** : intfloat/multilingual-e5-small
   - **Dimension** : 384 vecteurs
   - **Optimisation** : Batch processing pour la performance
   - **Stockage** : Format binaire optimisé

5. **Indexation FAISS :**
   - **Type d'index** : IndexFlatIP (Inner Product)
   - **Métrique** : Similarité cosinus
   - **Optimisation** : Quantisation pour réduire l'empreinte mémoire
   - **Persistance** : Sauvegarde sur disque

### 3.7.3 Qualité et validation des données

**Métriques de qualité :**
- **Complétude** : Couverture des modules académiques
- **Cohérence** : Uniformité du formatage
- **Actualité** : Fraîcheur des informations
- **Accessibilité** : Lisibilité et structure

**Processus de validation :**
- **Vérification automatique** : Scripts de validation
- **Contrôle manuel** : Échantillonnage et vérification
- **Feedback utilisateur** : Amélioration continue
- **Mise à jour** : Processus de rafraîchissement

## 3.8 INTÉGRATION ET ORCHESTRATION

### 3.8.1 Architecture système globale

**Composants principaux :**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Interface     │    │   API FastAPI   │    │   Pipeline      │
│   Utilisateur   │◄──►│   (Backend)     │◄──►│   KG-RAG-LLM    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Base de       │
                       │   Données       │
                       │   (FAISS + KG)  │
                       └─────────────────┘
```

**Flux de données :**
1. **Réception** : Question utilisateur via l'interface
2. **Prétraitement** : Nettoyage et normalisation
3. **Recherche KG** : Interrogation du graphe de connaissances
4. **Recherche RAG** : Récupération des documents pertinents
5. **Fusion** : Combinaison des résultats KG et RAG
6. **Génération** : Réponse par LLaMA 3.1
7. **Post-traitement** : Formatage et ajout des sources
8. **Retour** : Affichage à l'utilisateur

### 3.8.2 Gestion des erreurs et fallbacks

**Stratégies de fallback :**
- **Recherche KG échoue** : Utilisation exclusive du RAG
- **Recherche RAG échoue** : Réponse basée sur les connaissances générales
- **LLM indisponible** : Réponses pré-construites pour les questions fréquentes
- **Erreur de traitement** : Messages d'erreur informatifs

**Monitoring et logging :**
- **Logs détaillés** : Traçabilité complète des opérations
- **Métriques de performance** : Temps de réponse, taux de succès
- **Alertes** : Notification des erreurs critiques
- **Tableau de bord** : Visualisation des performances

## 3.9 MÉTRIQUES ET ÉVALUATION

### 3.9.1 Métriques techniques

**Performance système :**
- **Temps de réponse** : Objectif < 10 secondes
- **Débit** : Nombre de requêtes simultanées
- **Précision** : Objectif > 85%
- **Rappel** : Couverture des informations pertinentes

**Qualité des réponses :**
- **Similarité cosinus** : Mesure de pertinence
- **Cohérence** : Stabilité des réponses
- **Complétude** : Exhaustivité des informations
- **Traçabilité** : Qualité des citations

### 3.9.2 Métriques pédagogiques

**Efficacité d'apprentissage :**
- **Compréhension** : Amélioration de la compréhension
- **Engagement** : Temps d'interaction avec le système
- **Satisfaction** : Évaluation subjective de l'utilité
- **Transfert** : Application des connaissances

**Méthodes d'évaluation :**
- **Tests automatisés** : Validation des réponses
- **Évaluation humaine** : Jugement d'experts
- **Feedback utilisateur** : Questionnaires et entretiens
- **A/B testing** : Comparaison de versions

## 3.10 CONCLUSION

Ce chapitre a permis d'établir précisément les fondations conceptuelles et techniques de notre projet FSBM Scholar Assistant. Nous avons présenté les modèles de langage utilisés, justifié leur choix (LLaMA 3.1), détaillé notre approche hybride KG-RAG-LLM, et expliqué notre méthode pour préparer et exploiter les données dans le cadre de notre assistant académique intelligent.

**Points clés établis :**
- **Choix technologique** : LLaMA 3.1 via OpenRouter pour performance et accessibilité
- **Architecture innovante** : Combinaison synergique KG-RAG-LLM
- **Pipeline de données** : Traitement automatisé et optimisé des documents FSBM
- **Interface utilisateur** : Chatbot conversationnel et intuitif
- **Évaluation continue** : Métriques techniques et pédagogiques

**Prochaines étapes :**
Cette clarification approfondie des concepts et outils utilisés est essentielle avant d'aborder concrètement, dans le chapitre suivant, la conception et la réalisation effective du système complet. Nous avons maintenant une base solide pour implémenter notre solution et évaluer ses performances dans le contexte académique de la FSBM.

---

*Ce chapitre établit les fondements techniques et conceptuels de notre approche, démontrant la faisabilité et l'innovation de l'architecture KG-RAG-LLM proposée pour l'assistance académique intelligente.*
