# Pipeline Automatique d'Extraction de Concepts

## 🎯 Vue d'ensemble

Ce pipeline automatique surveille en permanence votre répertoire de documents et relance automatiquement l'extraction de concepts dès qu'un nouveau document PDF est détecté. Il fonctionne comme un système RAG intelligent qui s'adapte dynamiquement à vos données.

## ✨ Fonctionnalités

- **🔍 Surveillance automatique** : Détecte les nouveaux, modifiés et supprimés PDF
- **⚡ Extraction intelligente** : Relance automatiquement l'extraction de concepts
- **🔄 Mise à jour en temps réel** : Met à jour la base Neo4j automatiquement
- **📊 Sauvegarde intelligente** : Crée des sauvegardes horodatées
- **⚙️ Configuration flexible** : Paramètres personnalisables via JSON
- **📝 Logging complet** : Suivi détaillé de toutes les opérations

## 🚀 Installation

### 1. Dépendances requises

```bash
pip install -r pipeline_requirements.txt
```

### 2. Vérification de l'installation

```bash
python test_pipeline.py
```

## 📁 Structure des fichiers

```
knowledge_graph/
├── auto_concept_pipeline.py      # Pipeline principal
├── start_pipeline.py             # Script de démarrage
├── test_pipeline.py              # Tests du pipeline
├── pipeline_config.json          # Configuration
├── pipeline_requirements.txt     # Dépendances
├── dynamic_concept_extractor.py  # Extracteur de concepts
├── neo4j_manager.py             # Gestionnaire Neo4j
└── PIPELINE_README.md           # Ce fichier
```

## ⚙️ Configuration

### Fichier `pipeline_config.json`

```json
{
  "auto_extract": true,           # Extraction automatique activée
  "extraction_interval": 3600,    # Intervalle en secondes (1h)
  "watch_directory": true,        # Surveillance du répertoire
  "use_neo4j": true,             # Utilisation de Neo4j
  "backup_results": true,         # Sauvegarde des résultats
  "max_concepts_per_module": 100, # Limite de concepts par module
  "min_concept_frequency": 2      # Fréquence minimale des concepts
}
```

### Paramètres principaux

| Paramètre | Description | Valeur par défaut |
|-----------|-------------|-------------------|
| `auto_extract` | Extraction automatique | `true` |
| `extraction_interval` | Intervalle d'extraction (secondes) | `3600` (1h) |
| `watch_directory` | Surveillance des changements | `true` |
| `use_neo4j` | Intégration Neo4j | `true` |
| `backup_results` | Sauvegarde des résultats | `true` |

## 🎮 Utilisation

### 1. Démarrage simple

```bash
python start_pipeline.py
```

### 2. Options de démarrage

```bash
# Pipeline complet avec surveillance
python start_pipeline.py

# Extraction immédiate uniquement
python auto_concept_pipeline.py --extract-now

# Vérification du statut
python auto_concept_pipeline.py --status

# Pipeline avec répertoire personnalisé
python auto_concept_pipeline.py --docs /chemin/vers/documents
```

### 3. Menu interactif

Le script `start_pipeline.py` propose un menu avec 4 options :

1. **Pipeline complet** : Surveillance + extraction automatique
2. **Extraction immédiate** : Une seule extraction
3. **Vérifier le statut** : État actuel du pipeline
4. **Quitter** : Arrêt du programme

## 🔍 Fonctionnement

### Cycle de surveillance

1. **👀 Surveillance** : Le pipeline surveille le répertoire `../docs`
2. **📄 Détection** : Détecte les nouveaux/modifiés PDF
3. **⏰ Planification** : Planifie une extraction dans 30 secondes
4. **🔍 Extraction** : Lance l'extraction de concepts
5. **💾 Sauvegarde** : Sauvegarde les résultats avec timestamp
6. **🔄 Mise à jour** : Met à jour la base Neo4j si configurée
7. **📊 Logging** : Enregistre toutes les opérations

### Détection des changements

Le pipeline utilise un système de hash MD5 pour détecter les changements :

- **Hash des fichiers** : Nom + taille + date de modification
- **Comparaison** : Compare avec le hash de la dernière extraction
- **Déclenchement** : Relance l'extraction si changement détecté

## 📊 Sorties et résultats

### Fichiers générés

- `extraction_results_YYYYMMDD_HHMMSS.json` : Résultats horodatés
- `latest_extraction_results.json` : Derniers résultats
- `backup/extraction_results_*.json` : Sauvegardes
- `auto_pipeline.log` : Logs détaillés

### Structure des résultats

```json
{
  "concepts": {
    "java": ["servlet", "jsp", "jstl"],
    "bigdata": ["hadoop", "spark", "hive"]
  },
  "relations": [
    {"source": "servlet", "target": "jsp", "type": "uses"}
  ],
  "documents": [
    {"file": "document.pdf", "module": "java", "concepts_count": 15}
  ],
  "extraction_metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "documents_hash": "abc123...",
    "processing_time": 45.2,
    "total_documents": 25,
    "total_concepts": 150
  }
}
```

## 🛠️ Dépannage

### Problèmes courants

#### 1. Dépendances manquantes

```bash
# Solution
pip install -r pipeline_requirements.txt
```

#### 2. Répertoire des documents non trouvé

```bash
# Vérifier le chemin dans pipeline_config.json
# Créer le répertoire ../docs s'il n'existe pas
mkdir -p ../docs
```

#### 3. Erreur Neo4j

```bash
# Vérifier que Neo4j est démarré
# Vérifier la configuration dans pipeline_config.json
# Désactiver Neo4j temporairement : "use_neo4j": false
```

#### 4. Aucun concept extrait

```bash
# Vérifier que les PDF contiennent du texte
# Vérifier les paramètres d'extraction
# Consulter les logs pour plus de détails
```

### Logs et debugging

```bash
# Voir les logs en temps réel
tail -f auto_pipeline.log

# Vérifier le statut du pipeline
python auto_concept_pipeline.py --status

# Lancer les tests
python test_pipeline.py
```

## 🔧 Personnalisation avancée

### Ajouter de nouveaux modules

Dans `pipeline_config.json` :

```json
{
  "modules": {
    "nouveau_module": {
      "keywords": ["mot1", "mot2", "mot3"],
      "priority": 1
    }
  }
}
```

### Modifier les indicateurs techniques

```json
{
  "concept_extraction": {
    "technical_indicators": [
      "définition", "concept", "principe", "méthode",
      "votre_indicateur_personnalisé"
    ]
  }
}
```

### Configuration des notifications

```json
{
  "email_notifications": true,
  "email_config": {
    "smtp_server": "smtp.gmail.com",
    "sender_email": "votre@email.com",
    "recipient_emails": ["destinataire@email.com"]
  }
}
```

## 📈 Monitoring et métriques

### Métriques disponibles

- **Temps de traitement** : Durée de chaque extraction
- **Nombre de concepts** : Concepts extraits par module
- **Documents traités** : Nombre de PDF analysés
- **Fréquence d'extraction** : Nombre d'extractions par jour

### Surveillance en temps réel

```bash
# Voir le statut actuel
python auto_concept_pipeline.py --status

# Suivre les logs
tail -f auto_pipeline.log

# Vérifier les résultats récents
ls -la extraction_results_*.json
```

## 🚀 Intégration avec le chatbot

### Mise à jour automatique

Le pipeline met automatiquement à jour :

1. **Fichiers de concepts** : `latest_extraction_results.json`
2. **Base Neo4j** : Nœuds et relations des concepts
3. **Mots-clés** : Génération automatique des mots-clés

### Utilisation dans le chatbot

```python
# Charger les concepts extraits
with open("latest_extraction_results.json", "r") as f:
    concepts_data = json.load(f)

# Utiliser dans le chatbot
chatbot.update_concepts(concepts_data["concepts"])
```

## 🔒 Sécurité et bonnes pratiques

### Recommandations

1. **Sauvegarde régulière** : Gardez des copies des résultats
2. **Surveillance des logs** : Vérifiez régulièrement les logs
3. **Configuration sécurisée** : Protégez les fichiers de config
4. **Tests réguliers** : Lancez `test_pipeline.py` périodiquement

### Gestion des erreurs

Le pipeline gère automatiquement :

- **Erreurs de lecture PDF** : Continue avec les autres fichiers
- **Erreurs Neo4j** : Continue sans mise à jour de la base
- **Erreurs de sauvegarde** : Crée des sauvegardes alternatives
- **Timeouts** : Limite le temps de traitement

## 📚 Ressources et support

### Documentation

- `dynamic_concept_extractor.py` : Extracteur de concepts
- `neo4j_manager.py` : Gestionnaire de base de données
- `enhanced_kg_chatbot.py` : Chatbot avec knowledge graph

### Tests et validation

```bash
# Tests complets
python test_pipeline.py

# Test spécifique
python test_pipeline.py --help

# Validation des résultats
python -c "import json; data=json.load(open('latest_extraction_results.json')); print(f'Concepts: {len(data[\"concepts\"])}')"
```

### Support et maintenance

- **Logs** : Consultez `auto_pipeline.log`
- **Configuration** : Modifiez `pipeline_config.json`
- **Tests** : Utilisez `test_pipeline.py`
- **Statut** : Vérifiez avec `--status`

---

## 🎉 Félicitations !

Votre pipeline automatique d'extraction de concepts est maintenant configuré et prêt à fonctionner comme un système RAG intelligent qui s'adapte automatiquement à vos données !

**Prochaines étapes :**
1. ✅ Installez les dépendances
2. ✅ Configurez selon vos besoins
3. ✅ Testez le pipeline
4. 🚀 Lancez la surveillance automatique
5. 📚 Ajoutez vos documents PDF
6. 🔍 Laissez le pipeline extraire automatiquement les concepts !
