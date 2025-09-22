# CHAPITRE II : ÉTAT DE L'ART

## 2.1 INTRODUCTION

L'état de l'art constitue une analyse approfondie des technologies et des approches actuelles dans le domaine des systèmes d'intelligence artificielle intégrant les Graphes de Connaissances (Knowledge Graphs), le Retrieval Augmented Generation (RAG) et le Fine-Tuning de Large Language Models (LLMs). Dans le cadre de notre projet FSBM Scholar Assistant, maîtriser ces évolutions techniques récentes est fondamental, nous permettant d'intégrer les meilleures pratiques et d'assurer une réalisation scientifique rigoureuse et pertinente.

Cette démarche garantit non seulement une continuité avec les travaux existants, mais aussi une contribution originale en adéquation avec les dernières avancées technologiques. L'approche hybride KG-RAG-LLM que nous proposons représente une innovation dans le contexte académique marocain, combinant trois paradigmes complémentaires pour créer un assistant intelligent et contextuel.

## 2.2 GRAPHES DE CONNAISSANCES (KNOWLEDGE GRAPHS)

### 2.2.1 Concepts fondamentaux et architecture

Les Graphes de Connaissances représentent une approche structurée de la représentation des connaissances, où les entités (concepts, objets, personnes) sont connectées par des relations sémantiques explicites. Cette représentation graphique permet de modéliser la complexité des domaines académiques de manière intuitive et navigable.

**Architecture des graphes de connaissances :**
- **Nœuds (Entités)** : Concepts académiques, modules, enseignants, ressources
- **Arêtes (Relations)** : Précède, nécessite, enseigne, couvre, référence
- **Propriétés** : Métadonnées, scores de confiance, dates de mise à jour
- **Ontologies** : Taxonomies et schémas conceptuels du domaine

### 2.2.2 Applications dans l'éducation et l'assistance académique

**Systèmes existants :**
- **EduKG** : Graphe de connaissances éducatif pour l'enseignement supérieur
- **Academic Knowledge Graph** : Initiative de Google Scholar et Microsoft Academic
- **DBpedia** : Base de connaissances structurée pour l'éducation

**Avantages pour la FSBM :**
- **Navigation conceptuelle** : Relations explicites entre modules et concepts
- **Recherche contextuelle** : Compréhension des prérequis et dépendances
- **Recommandations intelligentes** : Suggestions de ressources pertinentes
- **Visualisation des parcours** : Cartographie des cursus académiques

### 2.2.3 Technologies et outils disponibles

**Plateformes de graphes :**
- **Neo4j** : Base de données graphe native avec langage Cypher
- **GraphDB** : Solution sémantique basée sur RDF/SPARQL
- **Amazon Neptune** : Service cloud pour graphes à grande échelle

**Outils d'extraction :**
- **OpenIE** : Extraction d'information ouverte
- **Stanford NLP** : Pipeline d'analyse linguistique
- **spaCy** : Traitement du langage naturel industriel

## 2.3 RETRIEVAL-AUGMENTED GENERATION (RAG)

### 2.3.1 Principes et architecture du RAG

Le paradigme RAG représente une évolution majeure dans la génération de contenu par IA, combinant la puissance des modèles de langage avec la précision de la récupération d'information. Cette approche hybride résout les limitations des LLMs classiques en matière de factualité et de traçabilité.

**Architecture RAG standard :**
```
[Question] → [Retriever] → [Documents] → [Generator] → [Réponse]
                ↓              ↓            ↓
            Embeddings    Base Vectorielle  LLM
            + Indexing    + Reranking     + Prompting
```

**Composants critiques :**
- **Retriever** : Système de récupération basé sur similarité sémantique
- **Reranker** : Affinement des résultats par pertinence contextuelle
- **Generator** : Modèle de langage pour la génération de réponses
- **Orchestrator** : Coordination et gestion des flux de données

### 2.3.2 Systèmes RAG existants et applications pratiques

**Plateformes commerciales :**
- **ChatGPT Plus** : Intégration RAG avec plugins et browsing
- **Claude Pro** : RAG avancé avec analyse de documents
- **Perplexity AI** : Recherche augmentée par IA

**Applications académiques :**
- **Elicit** : Assistant de recherche scientifique
- **Consensus** : Analyse de littérature académique
- **ScholarAI** : Assistant pour chercheurs et étudiants

**Avantages par rapport aux LLMs classiques :**
- **Factualité** : Réponses basées sur sources vérifiables
- **Traçabilité** : Citations et références explicites
- **Actualité** : Mise à jour continue des connaissances
- **Spécialisation** : Adaptation au domaine académique FSBM

### 2.3.3 Limitations et défis actuels

**Défis identifiés :**
- **Qualité des embeddings** : Représentation sémantique des concepts techniques
- **Reranking** : Ordre optimal des documents pour la génération
- **Hallucinations** : Génération d'informations non présentes dans les sources
- **Latence** : Temps de réponse pour les requêtes complexes

**Solutions proposées :**
- **Embeddings spécialisés** : Modèles adaptés au domaine scientifique
- **Reranking contextuel** : Évaluation de pertinence basée sur la question
- **Prompt engineering** : Instructions explicites pour éviter les hallucinations
- **Cache intelligent** : Mise en cache des requêtes fréquentes

## 2.4 FINE-TUNING DE MODÈLES DE LANGAGE

### 2.4.1 Concepts de base du transfer learning

Le Fine-Tuning représente une approche de transfer learning permettant d'adapter des modèles pré-entraînés à des domaines ou tâches spécifiques. Dans notre contexte académique, cette technique permet d'optimiser les performances sur le vocabulaire et les concepts de la FSBM.

**Types de fine-tuning :**
- **Full Fine-Tuning** : Adaptation complète de tous les paramètres
- **Parameter-Efficient Fine-Tuning (PEFT)** : Adaptation partielle (LoRA, AdaLoRA)
- **Instruction Tuning** : Adaptation aux formats de questions-réponses
- **Domain Adaptation** : Spécialisation au vocabulaire académique

### 2.4.2 Techniques de fine-tuning pour LLMs

**Méthodes PEFT :**
- **LoRA (Low-Rank Adaptation)** : Adaptation par matrices de rang faible
- **AdaLoRA** : Adaptation adaptative des rangs
- **QLoRA** : Quantisation pour réduire l'empreinte mémoire
- **Prefix Tuning** : Adaptation par préfixes d'instructions

**Avantages des approches PEFT :**
- **Efficacité mémoire** : Réduction drastique des ressources nécessaires
- **Rapidité** : Adaptation en quelques heures vs jours
- **Flexibilité** : Possibilité d'adapter plusieurs modèles simultanément
- **Réutilisabilité** : Adaptation incrémentale selon les besoins

### 2.4.3 Modèles LLaMA : architecture et spécificités

**Architecture LLaMA 3.1 :**
- **Transformer Decoder-Only** : Architecture unidirectionnelle optimisée
- **Attention optimisée** : Mécanismes d'attention avancés (Grouped Query Attention)
- **Context window** : Fenêtre contextuelle étendue (8K-32K tokens)
- **Efficacité** : Optimisations pour l'inférence et l'entraînement

**Spécificités pour l'éducation :**
- **Compréhension contextuelle** : Capacité à traiter des documents longs
- **Raisonnement** : Capacités de déduction et d'analyse
- **Multilingue** : Support français/anglais natif
- **Éthique** : Alignement avec les valeurs éducatives

### 2.4.4 Méthodes d'adaptation au domaine académique

**Stratégies d'adaptation :**
- **Corpus spécialisé** : Documents académiques FSBM
- **Prompt engineering** : Instructions spécifiques au domaine
- **Few-shot learning** : Exemples de questions-réponses
- **Chain-of-thought** : Raisonnement étape par étape

**Métriques d'évaluation :**
- **Perplexité** : Mesure de la qualité du modèle sur le domaine
- **Accuracy** : Précision des réponses générées
- **F1-Score** : Équilibre précision/rappel
- **ROUGE** : Évaluation de la qualité textuelle

## 2.5 CHATBOTS ÉDUCATIFS ET ASSISTANTS IA

### 2.5.1 État actuel des chatbots éducatifs

**Plateformes existantes :**
- **Duolingo** : Assistant linguistique avec IA conversationnelle
- **Khan Academy** : Tutorat intelligent adaptatif
- **Coursera** : Support étudiant automatisé
- **edX** : Assistant d'apprentissage personnalisé

**Fonctionnalités communes :**
- **Réponses contextuelles** : Adaptation au niveau de l'étudiant
- **Suivi de progression** : Monitoring des apprentissages
- **Ressources adaptatives** : Recommandations personnalisées
- **Support multilingue** : Assistance dans plusieurs langues

### 2.5.2 Comparaison des approches existantes

**Approches traditionnelles :**
- **Rule-based** : Réponses basées sur des règles prédéfinies
- **Template-based** : Génération à partir de modèles prédéfinis
- **FAQ systems** : Recherche dans des bases de questions-réponses

**Approches modernes :**
- **Neural chatbots** : Modèles de langage end-to-end
- **Hybrid systems** : Combinaison de règles et d'IA
- **RAG-based** : Récupération et génération augmentées

**Avantages de l'approche hybride :**
- **Robustesse** : Fallback sur des règles en cas d'échec de l'IA
- **Contrôle** : Maîtrise des réponses sensibles
- **Efficacité** : Optimisation des ressources de calcul
- **Maintenabilité** : Facilité de mise à jour et d'évolution

### 2.5.3 Gaps identifiés dans la littérature

**Limitations actuelles :**
- **Contexte académique** : Peu de systèmes spécialisés pour l'enseignement supérieur
- **Multidisciplinarité** : Difficulté à gérer plusieurs domaines simultanément
- **Évaluation continue** : Absence de métriques d'efficacité pédagogique
- **Intégration institutionnelle** : Défis d'intégration avec les systèmes existants

**Opportunités d'innovation :**
- **Graphes de connaissances** : Représentation structurée des relations académiques
- **RAG spécialisé** : Adaptation aux contenus scientifiques et techniques
- **Fine-tuning académique** : Optimisation pour le vocabulaire FSBM
- **Évaluation pédagogique** : Métriques d'efficacité d'apprentissage

### 2.5.4 Métriques d'évaluation pertinentes

**Métriques techniques :**
- **Temps de réponse** : Latence du système
- **Précision des réponses** : Exactitude des informations fournies
- **Pertinence** : Adéquation des réponses aux questions
- **Cohérence** : Stabilité des réponses sur des questions similaires

**Métriques pédagogiques :**
- **Compréhension** : Amélioration de la compréhension des étudiants
- **Engagement** : Temps d'interaction avec le système
- **Satisfaction** : Évaluation subjective de l'utilité
- **Transfert** : Application des connaissances dans d'autres contextes

## 2.6 INNOVATIONS ET CONTRIBUTIONS PROPOSÉES

### 2.6.1 Approche hybride KG-RAG-LLM

Notre projet propose une innovation majeure en combinant trois paradigmes complémentaires :

**Architecture hybride :**
```
[Question] → [KG Query] → [RAG Retrieval] → [LLM Generation] → [Réponse]
                ↓              ↓                ↓
            Graphe de      Base Vectorielle   Modèle Fine-tuné
            Connaissances  + Reranking       + Prompting
```

**Avantages de l'approche :**
- **Navigation conceptuelle** : Relations explicites entre concepts
- **Récupération précise** : Recherche sémantique optimisée
- **Génération contextuelle** : Réponses adaptées au domaine FSBM
- **Traçabilité complète** : Sources et relations explicites

### 2.6.2 Spécialisation pour le contexte FSBM

**Adaptations spécifiques :**
- **Ontologie académique** : Modélisation des cursus et modules
- **Vocabulaire technique** : Adaptation aux domaines BIGDATA, MNP, SMI
- **Relations pédagogiques** : Prérequis, co-requis, dépendances
- **Métadonnées enrichies** : Niveaux, crédits, objectifs d'apprentissage

### 2.6.3 Contribution à la recherche en IA éducative

**Innovations méthodologiques :**
- **Pipeline d'extraction** : Méthodes automatisées de construction de graphes
- **Évaluation hybride** : Métriques combinant aspects techniques et pédagogiques
- **Benchmark académique** : Dataset de référence pour l'évaluation
- **Framework open-source** : Outils reproductibles pour la communauté

## 2.7 CONCLUSION

Ce chapitre présente les bases techniques solides nécessaires à notre projet FSBM Scholar Assistant, tout en soulignant les choix méthodologiques et technologiques adoptés. L'état de l'art révèle un domaine en pleine évolution, avec des opportunités significatives d'innovation dans l'assistance académique.

**Synthèse des choix technologiques :**
- **Graphes de connaissances** : Représentation structurée des relations académiques
- **Architecture RAG** : Récupération et génération augmentées pour la précision
- **Fine-tuning LLaMA** : Adaptation au domaine académique FSBM
- **Approche hybride** : Combinaison synergique des trois technologies

**Contributions originales :**
- **Première application KG-RAG-LLM** dans le contexte académique marocain
- **Ontologie spécialisée** pour les filières FSBM
- **Métriques d'évaluation** combinant aspects techniques et pédagogiques
- **Framework reproductible** pour la communauté de recherche

En intégrant ces technologies avancées avec une approche hybride innovante, nous sommes en mesure de produire un assistant académique performant, fiable et contextuellement adapté, répondant parfaitement aux exigences académiques et pédagogiques définies dans notre cahier des charges. Cette approche positionne notre projet à l'avant-garde de l'innovation en IA éducative, tout en offrant une solution pratique et efficace pour les défis spécifiques de la FSBM.

---

*Ce chapitre établit les fondements théoriques et technologiques de notre approche, démontrant la pertinence et l'innovation de l'architecture KG-RAG-LLM proposée pour l'assistance académique intelligente.*
