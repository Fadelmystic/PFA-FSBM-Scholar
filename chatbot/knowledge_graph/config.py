import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Extraction
MIN_CONFIDENCE = 0.6
MAX_ENTITIES_PER_DOC = 50

# Types d'entités de base
ENTITY_TYPES = [
    "Cours", "Chapitre", "Concept", "Personne", "Ressource", "Langage"
]

# Types de relations de base
RELATION_TYPES = [
    "FAIT_PARTIE_DE", "ENSEIGNE_PAR", "DEFINI_PAR", "LIE_A", "PREREQUIS",
    "DÉFINI_DANS", "UTILISÉ_DANS", "A_POUR_TYPE"
]


