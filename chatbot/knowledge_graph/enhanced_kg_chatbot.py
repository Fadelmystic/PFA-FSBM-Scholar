#!/usr/bin/env python3
"""
Enhanced KG Chatbot for FSBM Scholar Assistant
- FastAPI server
- Concept extraction (lightweight)
- Neo4j querying for definitions/examples/properties/relations
- Unified, structured formatting via format_kg_response
"""

from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import re

from neo4j_manager import Neo4jManager
from kg_api import format_kg_response, generate_kg_response, initialize_kg


class ChatRequest(BaseModel):
	message: str


class ChatResponse(BaseModel):
	response: str
	entities: List[Dict]
	relations: List[Dict]
	confidence: float


app = FastAPI(title="FSBM Scholar Assistant - Enhanced KG", version="2.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

db: Optional[Neo4jManager] = None


@app.on_event("startup")
async def on_start() -> None:
	global db
	db = Neo4jManager()
	# Initialiser le KG dans kg_api pour activer le formatage LLM
	try:
		initialize_kg()
	except Exception:
		pass


@app.on_event("shutdown")
async def on_stop() -> None:
	if db:
		db.close()


@app.get("/health")
async def health():
	if db and db.test_connection():
		return {"status": "healthy", "version": app.version}
	return {"status": "unhealthy"}


class EnhancedKGQueryEngine:
	def __init__(self, db: Neo4jManager):
		self.db = db
		self.concept_keywords: Dict[str, List[str]] = {
			'java': ['java', 'jdk', 'jre', 'jvm', 'servlet', 'jsp', 'jstl', 'jdbc', 'jpa', 'jee'],
			'poo': ['poo', 'oop', 'classe', 'objet', 'héritage', 'heritage', 'polymorphisme', 'encapsulation', 'abstraction', 'composition'],
			'bigdata': ['big data', 'bigdata', 'hadoop', 'hdfs', 'mapreduce', 'yarn', 'spark', 'hive', 'hbase'],
			'ml': ['machine learning', 'ml', 'régression', 'classification', 'clustering'],
			'cloud': ['cloud', 'aws', 'azure', 'gcp'],
			'crypto': ['cryptographie', 'blockchain', 'bitcoin', 'hash'],
			'bdd': ['base de données', 'database', 'sql', 'oracle', 'mysql', 'mongodb', 'neo4j']
		}

	def detect_query_type(self, message: str) -> str:
		msg = message.lower()
		if any(k in msg for k in ['différence', 'différences', 'comparer', 'comparaison', 'vs', 'versus', 'entre', 'contrairement']):
			return 'comparison'
		if any(k in msg for k in ["explique", "définis", "definition", "définition", "qu'est-ce", "qu\"est-ce"]):
			return 'definition'
		if any(k in msg for k in ['exemple', 'exemples', 'comment', 'comment faire']):
			return 'example'
		if any(k in msg for k in ['module', 'cours', 'enseigné']):
			return 'module_info'
		return 'general'

	def extract_concepts(self, message: str) -> List[str]:
		msg = message.lower()
		found: List[str] = []
		stop_words = {
			"explique", "expliquer", "definition", "définition", "definir", "définir",
			"qu'est-ce", "qu\"est-ce", "entre", "différence", "différences", "et",
			"de", "du", "la", "le", "les", "un", "une", "des", "en"
		}
		for _, keywords in self.concept_keywords.items():
			for kw in keywords:
				if kw in msg:
					found.append(kw)
		# Fallback simple: mots alphanumériques significatifs
		if not found:
			candidates = re.findall(r"[a-zA-ZÀ-ÿ0-9\-\+]{3,}", msg)
			found = [c for c in candidates if (not c.isdigit()) and (c not in stop_words)]
		return list(dict.fromkeys(found))  # unique, keep order

	def query_definition(self, concepts: List[str]) -> str:
		if not concepts:
			return "Aucun concept détecté. Reformulez votre question."
		parts: List[str] = []
		for concept in concepts[:3]:
			q = f"""
			MATCH (d:Definition)-[:LIE_A]->(c:Concept)
			WHERE toLower(c.nom) CONTAINS toLower('{concept}')
			RETURN c.nom as concept, d.nom as definition
			LIMIT 3
			"""
			rows = self.db.query(q)
			if rows:
				parts.append(f"📖 **Définition de {rows[0]['concept']} :**\n{rows[0]['definition']}")
			else:
				parts.append(f"📖 **Définition de {concept} :**\nAucune définition trouvée dans le graphe.\n")
		return "\n\n".join(parts)

	def query_examples(self, concepts: List[str]) -> str:
		if not concepts:
			return "Aucun concept détecté."
		parts: List[str] = []
		for concept in concepts[:3]:
			q = f"""
			MATCH (c:Concept)-[:A_POUR_EXEMPLE]->(e:Exemple)
			WHERE toLower(c.nom) CONTAINS toLower('{concept}')
			RETURN c.nom as concept, e.nom as exemple
			LIMIT 5
			"""
			rows = self.db.query(q)
			if rows:
				parts.append(f"💡 **Exemples pour {rows[0]['concept']} :**")
				for r in rows:
					parts.append(f"   • {r['exemple']}")
			else:
				parts.append(f"❌ Aucun exemple trouvé pour '{concept}'.")
		return "\n\n".join(parts)

	def query_comparison(self, concepts: List[str]) -> str:
		if len(concepts) < 2:
			return "Pour une comparaison, mentionnez au moins deux concepts."
		c1, c2 = concepts[0], concepts[1]
		q = f"""
		MATCH (a:Concept)-[r]->(b:Concept)
		WHERE (toLower(a.nom) CONTAINS toLower('{c1}') AND toLower(b.nom) CONTAINS toLower('{c2}'))
		   OR (toLower(a.nom) CONTAINS toLower('{c2}') AND toLower(b.nom) CONTAINS toLower('{c1}'))
		RETURN a.nom as c1, type(r) as relation, b.nom as c2, r.context as context
		LIMIT 6
		"""
		rows = self.db.query(q)
		parts: List[str] = [f"🔍 **Comparaison entre {c1} et {c2} :**"]
		if rows:
			parts.append("🔗 **Relations trouvées :**")
			for r in rows:
				parts.append(f"   • {r['c1']} [{r['relation']}] → {r['c2']}")
				if r.get('context'):
					parts.append(f"     *{r['context']}*")
		else:
			parts.append("❌ Aucune relation directe trouvée dans le graphe.")
		return "\n\n".join(parts)

	def query_module_info(self, concepts: List[str]) -> str:
		if not concepts:
			return "Précisez un concept pour récupérer les modules/cours associés."
		concept = concepts[0]
		q = f"""
		MATCH (co:Cours)-[:LIE_A]->(c:Concept)
		WHERE toLower(c.nom) CONTAINS toLower('{concept}')
		RETURN c.nom as concept, co.nom as cours
		LIMIT 5
		"""
		rows = self.db.query(q)
		if rows:
			parts = [f"📚 **Cours associés à {rows[0]['concept']} :**"]
			for r in rows:
				parts.append(f"   • {r['cours']}")
			return "\n".join(parts)
		return "Aucun cours associé trouvé."

	def get_relations_for_concepts(self, concepts: List[str]) -> List[Dict]:
		if not concepts:
			return []
		names_lower = [c.lower() for c in concepts]
		lst = "[" + ",".join([f"'{x}'" for x in names_lower]) + "]"
		q = f"""
		MATCH (c1:Concept)-[r]->(c2:Concept)
		WHERE toLower(c1.nom) IN {lst} OR toLower(c2.nom) IN {lst}
		RETURN c1.nom as source, type(r) as relation, c2.nom as target, r.context as context
		LIMIT 20
		"""
		rows = self.db.query(q)
		rels: List[Dict] = []
		for row in rows or []:
			rels.append({
				"source": row.get("source", ""),
				"relation": row.get("relation", ""),
				"target": row.get("target", ""),
				"context": row.get("context", "") or ""
			})
		return rels

	def _get_concept_module(self, concept: str) -> str:
		m = {
			'java': 'JAVA', 'jdk': 'JAVA', 'jre': 'JAVA', 'jvm': 'JAVA',
			'servlet': 'JEE', 'jsp': 'JEE', 'jstl': 'JEE', 'jdbc': 'JEE', 'jpa': 'JEE', 'jee': 'JEE',
			'héritage': 'JAVA', 'heritage': 'JAVA', 'composition': 'JAVA', 'polymorphisme': 'JAVA', 'encapsulation': 'JAVA', 'constructeur': 'JAVA',
			'hadoop': 'BIGDATA', 'hdfs': 'BIGDATA', 'mapreduce': 'BIGDATA', 'yarn': 'BIGDATA', 'spark': 'BIGDATA', 'hive': 'BIGDATA', 'hbase': 'BIGDATA',
			'machine learning': 'ML', 'ml': 'ML', 'régression': 'ML', 'classification': 'ML',
			'blockchain': 'CRYPTO', 'cryptographie': 'CRYPTO', 'hash': 'CRYPTO',
			'sql': 'BDD', 'database': 'BDD', 'mongodb': 'BDD', 'oracle': 'BDD'
		}
		return m.get(concept.lower(), 'AUTRE')

	def _get_concept_definition(self, concept: str) -> str:
		defs = {
			"héritage": "Mécanisme permettant d'hériter attributs/méthodes d'une autre classe",
			"heritage": "Mécanisme permettant d'hériter attributs/méthodes d'une autre classe",
			"composition": "Mécanisme pour composer une classe d'instances d'autres classes",
			"polymorphisme": "Capacité d'avoir plusieurs comportements via une même interface",
			"encapsulation": "Masquage/contrôle d'accès aux données et méthodes",
			"constructeur": "Méthode spéciale exécutée à la création d'un objet",
			"java": "Langage orienté objet portable (JVM)",
			"hadoop": "Framework distribué pour gros volumes de données",
			"machine learning": "Apprentissage automatique à partir des données",
			"blockchain": "Registre distribué immuable et sécurisé",
			"sql": "Langage de requêtes pour bases de données relationnelles",
			"cloud": "Fourniture de services informatiques via Internet (IaaS, PaaS, SaaS)"
		}
		return defs.get(concept.lower(), "Concept technique extrait de la base de connaissances")

	def _get_concept_examples(self, concept: str) -> List[str]:
		ex = {
			"héritage": ["extends Animal", "super()"],
			"composition": ["private Moteur moteur", "has-a"],
			"polymorphisme": ["Animal a = new Chien()", "override"],
			"encapsulation": ["private", "getters/setters"],
			"java": ["System.out.println()", "public class"],
			"hadoop": ["HDFS", "MapReduce"],
			"machine learning": ["Régression", "Classification"],
			"blockchain": ["Smart contracts", "SHA-256"],
			"sql": ["SELECT * FROM ...", "CREATE TABLE"],
			"cloud": ["IaaS / PaaS / SaaS", "AWS Lambda", "Kubernetes"]
		}
		return ex.get(concept.lower(), [f"Exemple d'utilisation de {concept}"])

	def _get_concept_properties(self, concept: str) -> List[str]:
		props = {
			"héritage": ['Relation "est-un"', 'extends'],
			"composition": ['Relation "a-un"', 'flexible'],
			"polymorphisme": ['Override', 'Interface commune'],
			"encapsulation": ['Private/Public/Protected'],
			"constructeur": ['Nom de la classe', 'Pas de type de retour'],
			"java": ['Orienté objet', 'Portable (JVM)'],
			"hadoop": ['Distribué', 'Tolérant aux pannes'],
			"machine learning": ['Prédiction', 'Généralisation'],
			"blockchain": ['Décentralisé', 'Immuable'],
			"sql": ['ACID', 'Relationnel'],
			"cloud": ['Scalabilité', 'Pay-per-use', 'Haute disponibilité']
		}
		return props.get(concept.lower(), [f"Propriété de {concept}"])

	def _get_rich_content(self) -> Dict[str, str]:
		"""Contenu enrichi pré-structuré pour des réponses rapides"""
		return {
			"constructeur": """
DEFINITION
Un constructeur est une méthode spéciale appelée lors de la création d'un objet pour initialiser ses attributs.

CARACTERISTIQUES PRINCIPALES
- Même nom que la classe
- Pas de type de retour (même pas void)
- Appel automatique avec 'new'
- Peut être surchargé (plusieurs constructeurs)

EXEMPLE PRATIQUE
```java
public class Etudiant {
    private String nom;
    private int age;
    
    // Constructeur paramétré
    public Etudiant(String nom, int age) {
        this.nom = nom;
        this.age = age;
    }
    
    // Constructeur par défaut
    public Etudiant() {
        this("Inconnu", 0);
    }
}
```

TYPES DE CONSTRUCTEURS
- Par défaut : Généré automatiquement si aucun constructeur défini
- Paramétré : Avec paramètres pour initialiser les attributs
- En chaîne : Constructeurs qui s'appellent mutuellement avec this()

AVANTAGES
- Initialisation automatique des objets
- Encapsulation des paramètres d'initialisation
- Surcharge pour différents cas d'usage
- Validation des données à la création
""",
			"cloud": """
DEFINITION
Le Cloud Computing est la fourniture de services informatiques via Internet.

MODELES DE SERVICE
- IaaS (Infrastructure as a Service) : Serveurs, stockage, réseau
- PaaS (Platform as a Service) : Environnement de développement
- SaaS (Software as a Service) : Applications prêtes à l'emploi

FOURNISSEURS PRINCIPAUX
- AWS (Amazon Web Services)
- Azure (Microsoft)
- GCP (Google Cloud Platform)

EXEMPLE D'UTILISATION
```yaml
# Déploiement cloud avec Docker
version: '3'
services:
  app:
    image: mon-app:latest
    ports:
      - "8080:8080"
    environment:
      - DB_HOST=postgres
```

AVANTAGES
- Scalabilité automatique
- Coûts optimisés (pay-per-use)
- Haute disponibilité
- Déploiement global
- Maintenance simplifiée
- Sécurité renforcée
""",
			"héritage": """
DEFINITION
L'héritage permet à une classe d'hériter des propriétés et méthodes d'une autre classe.

RELATION
"est-un" (is-a)

MOT-CLE
extends

EXEMPLE
```java
public class Animal {
    protected String nom;
    public void faireBruit() { 
        System.out.println("Bruit"); 
    }
}

public class Chien extends Animal {
    public void aboyer() { 
        System.out.println("Wouf!"); 
    }
}
```

AVANTAGES
- Réutilisation du code
- Extension de fonctionnalités
- Hiérarchie de classes
- Polymorphisme
- Organisation logique du code
""",
			"composition": """
DEFINITION
La composition permet à une classe de contenir des instances d'autres classes.

RELATION
"a-un" (has-a)

AVANTAGE
Plus flexible que l'héritage

EXEMPLE
```java
public class Moteur {
    public void demarrer() { 
        System.out.println("Vroum!"); 
    }
}

public class Voiture {
    private Moteur moteur;  // Composition
    
    public Voiture() {
        this.moteur = new Moteur();
    }
    
    public void demarrer() {
        moteur.demarrer();
    }
}
```

AVANTAGES
- Flexibilité élevée
- Contrôle du cycle de vie
- Évite l'héritage multiple
- Couplage faible
- Réutilisabilité des composants
""",
			"blockchain": """
DEFINITION
La blockchain est un registre distribué et décentralisé qui enregistre les transactions de manière sécurisée et immuable.

CARACTERISTIQUES PRINCIPALES
- Décentralisation : Pas d'autorité centrale
- Immutabilité : Les données ne peuvent pas être modifiées
- Transparence : Toutes les transactions sont visibles
- Sécurité : Cryptographie avancée

TYPES DE BLOCKCHAIN
- Publique : Bitcoin, Ethereum
- Privée : Hyperledger, Corda
- Consortium : Plusieurs organisations

EXEMPLE D'UTILISATION
```javascript
// Smart Contract Ethereum
contract SimpleStorage {
    uint256 private storedData;
    
    function set(uint256 x) public {
        storedData = x;
    }
    
    function get() public view returns (uint256) {
        return storedData;
    }
}
```

AVANTAGES
- Sécurité élevée
- Transparence totale
- Pas d'intermédiaire
- Traçabilité complète
- Résistance à la censure
""",
			"hadoop": """
DEFINITION
Hadoop est un framework open-source pour le traitement distribué de gros volumes de données.

COMPOSANTS PRINCIPAUX
- HDFS (Hadoop Distributed File System) : Stockage distribué
- MapReduce : Modèle de programmation pour traitement parallèle
- YARN : Gestionnaire de ressources
- Hive : Data warehouse pour requêtes SQL

EXEMPLE MAPREDUCE
```java
public class WordCount {
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] words = value.toString().split(" ");
            for (String word : words) {
                context.write(new Text(word), new IntWritable(1));
            }
        }
    }
}
```

AVANTAGES
- Scalabilité horizontale
- Tolérance aux pannes
- Coût réduit
- Flexibilité du stockage
- Traitement parallèle
""",
			"machine learning": """
DEFINITION
Le Machine Learning est une branche de l'intelligence artificielle qui permet aux systèmes d'apprendre automatiquement à partir de données.

TYPES D'APPRENTISSAGE
- Supervisé : Avec données étiquetées
- Non supervisé : Sans données étiquetées
- Par renforcement : Apprentissage par essai-erreur

ALGORITHMES POPULAIRES
- Régression linéaire
- Classification (SVM, Random Forest)
- Clustering (K-means)
- Réseaux de neurones

EXEMPLE PYTHON
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Données d'entraînement
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Modèle
model = LinearRegression()
model.fit(X, y)

# Prédiction
prediction = model.predict([[6]])
print(f"Prédiction: {prediction[0]}")
```

AVANTAGES
- Automatisation des tâches
- Prédictions précises
- Découverte de patterns
- Optimisation continue
- Personnalisation
"""
		}

	def _build_structured_kg_section(self, kg_concepts: List[Dict], relations: List[Dict], details: Dict[str, Dict]) -> str:
		"""Construit une section KG structurée et rapide"""
		lines = []
		lines.append("RESULTATS KNOWLEDGE GRAPH")
		lines.append(f"Confiance globale : {95.0}%")
		
		if kg_concepts:
			lines.append("\nCONCEPTS DETECTES")
			for c in kg_concepts:
				concept_name = c.get("name", "Inconnu")
				module = c.get("module", "N/A")
				lines.append(f"- {concept_name} (module : {module})")
				
				# Détails enrichis
				concept_lower = concept_name.lower()
				if concept_lower in details:
					detail_info = details[concept_lower]
					if "definition" in detail_info:
						lines.append(f"  Definition : {detail_info['definition']}")
					if "examples" in detail_info:
						examples = detail_info['examples']
						if isinstance(examples, list):
							examples_str = ', '.join(examples[:2])
						else:
							examples_str = str(examples)
						lines.append(f"  Exemples : {examples_str}")
					if "properties" in detail_info:
						props = detail_info['properties']
						if isinstance(props, list):
							props_str = ', '.join(props[:2])
						else:
							props_str = str(props)
						lines.append(f"  Proprietes : {props_str}")
		
		if relations:
			lines.append("\nRELATIONS ASSOCIEES")
			# Limiter à 5 relations les plus pertinentes
			for r in relations[:5]:
				src = r.get("source", "?")
				rel = r.get("relation", "?")
				tgt = r.get("target", "?")
				lines.append(f"- {src} [{rel}] -> {tgt}")
		
		return "\n".join(lines)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
	if not db:
		raise HTTPException(500, "Service non initialisé")

	engine = EnhancedKGQueryEngine(db)
	query_type = engine.detect_query_type(req.message)
	concepts = engine.extract_concepts(req.message)

	# Content response according to type
	if query_type == 'definition':
		content = engine.query_definition(concepts)
	elif query_type == 'example':
		content = engine.query_examples(concepts)
	elif query_type == 'comparison':
		content = engine.query_comparison(concepts)
	elif query_type == 'module_info':
		content = engine.query_module_info(concepts)
	else:
		content = engine.query_definition(concepts) if concepts else (
			"""
ASSISTANT FSBM SCHOLAR - KG AMELIORE

Je peux vous aider avec :
- Java/JEE, Big Data, ML, Crypto/Blockchain, Bases de données
- Definitions, exemples, comparaisons, infos de modules

Essayez par ex. :
- "explique heritage en java"
- "difference entre heritage et composition"
- "exemples de hadoop"
			"""
		)

	# Prepare KG concepts + details
	kg_concepts: List[Dict] = [{
		"name": c,
		"module": engine._get_concept_module(c),
		"type": "Concept"
	} for c in concepts]

	details: Dict[str, Dict] = {}
	for c in concepts:
		lc = c.lower()
		details[lc] = {
			"definition": engine._get_concept_definition(c),
			"examples": engine._get_concept_examples(c),
			"properties": engine._get_concept_properties(c)
		}

	# Relations
	relations = engine.get_relations_for_concepts(concepts)

	# Réponse avec LLM pour une meilleure qualité
	if concepts:
		try:
			# Essayer d'utiliser la génération LLM structurée via kg_api
			final_response = generate_kg_response(req.message, kg_concepts)
		except Exception as e:
			# Fallback: réponse structurée locale
			response_parts = []
			
			# Section principale avec contenu enrichi
			if content and "Aucune définition trouvée" not in content:
				response_parts.append(content)
			else:
				# Contenu enrichi basé sur les concepts détectés
				for concept in concepts:
					concept_lower = concept.lower()
					if concept_lower in engine._get_rich_content():
						response_parts.append(engine._get_rich_content()[concept_lower])
			
			# Section KG structurée
			kg_section = engine._build_structured_kg_section(kg_concepts, relations, details)
			response_parts.append(kg_section)
			
			final_response = "\n\n".join(response_parts)
	else:
		final_response = content

	confidence = 0.9 if concepts else 0.3
	return ChatResponse(
		response=final_response,
		entities=[{"type": "Concept", "nom": c} for c in concepts],
		relations=relations,
		confidence=confidence
	)


if __name__ == "__main__":
	print("🚀 Démarrage du chatbot KG amélioré...")
	print("📡 API disponible sur http://localhost:8001")
	uvicorn.run(app, host="0.0.0.0", port=8001)
