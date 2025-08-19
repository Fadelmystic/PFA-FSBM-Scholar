#!/usr/bin/env python3
"""
API simple pour le Knowledge Graph
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j_manager import Neo4jManager
import logging
import os
import requests
from langchain.llms.base import LLM
from typing import Dict, Any, List, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin

# Initialiser Neo4j
neo4j_manager = None

# Configuration OpenRouter (comme dans le RAG)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-405b-instruct:free")

class OpenRouterLLM(LLM):
    """Wrapper LangChain pour l'API OpenRouter (comme dans le RAG)"""
    
    api_key: str
    model: str = "meta-llama/llama-3.1-405b-instruct:free"
    temperature: float = 0.35

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Erreur OpenRouter: {e}")
            return f"Erreur lors de la génération de la réponse: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API KG"""
    try:
        if neo4j_manager:
            stats = neo4j_manager.get_statistics()
            return jsonify({
                "status": "healthy",
                "neo4j_connected": True,
                "concepts_count": stats.get('concepts_count', 0),
                "relations_count": stats.get('relations_count', 0),
                "modules": stats.get('modules', [])
            })
        else:
            return jsonify({
                "status": "warning",
                "neo4j_connected": False,
                "message": "Neo4j non connecté"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint principal pour le chat KG (compatible avec le front-end)"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Message vide"}), 400
        
        logger.info(f"Message KG reçu: {user_message}")
        
        # Rechercher des concepts pertinents
        relevant_concepts = search_kg_concepts(user_message)
        
        # Générer une réponse basée sur les concepts trouvés
        response = generate_kg_response(user_message, relevant_concepts)
        
        return jsonify({
            "response": response,
            "concepts_found": relevant_concepts,
            "source": "Knowledge Graph"
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/kg/query', methods=['POST'])
def kg_query():
    """Endpoint principal pour les requêtes KG"""
    try:
        data = request.get_json()
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "Requête vide"}), 400
        
        logger.info(f"Requête KG reçue: {user_query}")
        
        # Rechercher des concepts pertinents
        relevant_concepts = search_kg_concepts(user_query)
        
        # Générer une réponse basée sur les concepts trouvés
        response = generate_kg_response(user_query, relevant_concepts)
        
        return jsonify({
            "query": user_query,
            "response": response,
            "concepts_found": relevant_concepts,
            "source": "Knowledge Graph"
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /kg/query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/kg/concepts', methods=['GET'])
def get_kg_concepts():
    """Récupérer tous les concepts du KG"""
    try:
        if not neo4j_manager:
            return jsonify({"error": "Neo4j non connecté"}), 500
        
        # Récupérer les concepts par module
        modules = ['BIGDATA', 'SMI', 'JAVA', 'JEE']
        all_concepts = {}
        
        for module in modules:
            concepts = neo4j_manager.get_concepts_by_module(module)
            all_concepts[module] = [c['nom'] for c in concepts]
        
        return jsonify(all_concepts)
        
    except Exception as e:
        logger.error(f"Erreur dans /kg/concepts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/kg/search', methods=['POST'])
def search_kg_concepts():
    """Rechercher des concepts dans le KG"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "Requête vide"}), 400
        
        concepts = search_kg_concepts(query)
        
        return jsonify({
            "query": query,
            "concepts": concepts,
            "count": len(concepts)
        })
        
    except Exception as e:
        logger.error(f"Erreur dans /kg/search: {e}")
        return jsonify({"error": str(e)}), 500

def search_kg_concepts(query):
    """Rechercher des concepts pertinents dans le KG"""
    try:
        if not neo4j_manager:
            return []
        
        # Recherche simple par mot-clé
        query_lower = query.lower()
        all_concepts = neo4j_manager.get_all_concepts()
        
        relevant = []
        for concept in all_concepts:
            if concept['nom'] and (query_lower in concept['nom'].lower() or 
                                 concept['nom'].lower() in query_lower):
                relevant.append({
                    "name": concept['nom'],
                    "module": concept['module']
                })
        
        return relevant[:10]  # Limiter à 10 résultats
        
    except Exception as e:
        logger.error(f"Erreur recherche concepts KG: {e}")
        return []

def format_kg_response(concepts: List[Dict], relations: List[Dict] = None, confidence: float = 0.8, details: Optional[Dict] = None) -> str:
    """
    Formate la réponse du KG en un texte structuré et lisible.
    - concepts : liste de dicts avec {name, module, ...}
    - relations : liste de dicts avec {source, relation, target, context}
    - confidence : score global de confiance
    - details : infos supplémentaires par concept {name: {definition, examples, ...}}
    """
    
    lines = []
    lines.append("📊 **Résultats Knowledge Graph**")
    lines.append(f"🔹 Confiance globale : **{confidence*100:.1f}%**")
    
    # Section Concepts
    if concepts:
        lines.append("\n🧠 **Concepts détectés :**")
        for c in concepts:
            concept_name = c.get("name", "Inconnu")
            module = c.get("module", "N/A")
            lines.append(f"• **{concept_name}** _(module : {module})_")
            
            # Ajouter les détails si disponibles
            if details and concept_name.lower() in details:
                detail_info = details[concept_name.lower()]
                if "definition" in detail_info:
                    lines.append(f"   📖 **Définition :** {detail_info['definition']}")
                if "examples" in detail_info:
                    examples = detail_info['examples']
                    if isinstance(examples, list):
                        examples_str = ', '.join(examples[:3])  # Limiter à 3 exemples
                    else:
                        examples_str = str(examples)
                    lines.append(f"   💡 **Exemples :** {examples_str}")
                if "properties" in detail_info:
                    props = detail_info['properties']
                    if isinstance(props, list):
                        props_str = ', '.join(props[:3])  # Limiter à 3 propriétés
                    else:
                        props_str = str(props)
                    lines.append(f"   ⚙️ **Propriétés :** {props_str}")
    else:
        lines.append("\n⚠️ Aucun concept détecté dans la requête.")
    
    # Section Relations
    if relations:
        lines.append("\n🔗 **Relations associées :**")
        for r in relations:
            src = r.get("source", "?")
            rel = r.get("relation", "?")
            tgt = r.get("target", "?")
            ctx = r.get("context", "")
            lines.append(f"• {src} **[{rel}]** → {tgt}")
            if ctx:
                lines.append(f"   📌 {ctx}")
    else:
        lines.append("\n⚠️ Aucune relation trouvée entre ces concepts.")
    
    return "\n".join(lines)

def generate_kg_response(user_query, concepts):
    """Générer une réponse structurée avec le modèle de langage (comme le RAG)"""
    if not concepts:
        return f"Je n'ai pas trouvé de concepts spécifiques pour '{user_query}' dans le Knowledge Graph. Essayez de reformuler votre question ou consultez la liste des concepts disponibles."
    
    try:
        # Initialiser le modèle de langage
        if not OPENROUTER_API_KEY:
            return generate_enhanced_fallback_response(user_query, concepts)
        
        llm = OpenRouterLLM(
            api_key=OPENROUTER_API_KEY,
            model=OPENROUTER_MODEL,
        )
        
        # Préparer le contexte avec les concepts trouvés
        concept_names = [c['name'] for c in concepts]
        modules = list(set([c['module'] for c in concepts if c.get('module')]))
        
        # Créer un prompt structuré comme dans le RAG
        context = f"Concepts trouvés dans le Knowledge Graph: {', '.join(concept_names[:10])}\n"
        context += f"Modules concernés: {', '.join(modules)}\n"
        
        prompt = f"""Tu es un assistant spécialisé en informatique et technologies. 
        
        Question de l'utilisateur: {user_query}
        
        Contexte du Knowledge Graph:
        {context}
        
        Génère une réponse structurée et détaillée basée sur ces concepts. 
        Organise ta réponse avec des points clés, des exemples et des explications techniques.
        Sois précis et utilise le vocabulaire technique approprié.
        
        Réponse:"""
        
        # Générer la réponse avec le modèle
        response = llm._call(prompt)
        
        # Formater avec la nouvelle fonction
        # Enrichir les relations et détails depuis Neo4j si possible
        details: Dict[str, Dict] = {}
        relations: List[Dict] = []
        try:
            if neo4j_manager:
                # Relations
                names_lower = [n.lower() for n in concept_names]
                list_str = "[" + ",".join([f"'{x}'" for x in names_lower]) + "]"
                rel_query = f"""
                MATCH (c1:Concept)-[r]->(c2:Concept)
                WHERE toLower(c1.nom) IN {list_str} OR toLower(c2.nom) IN {list_str}
                RETURN c1.nom as source, type(r) as relation, c2.nom as target, r.context as context
                LIMIT 20
                """
                rel_results = neo4j_manager.query(rel_query)
                for row in rel_results or []:
                    relations.append({
                        'source': row.get('source', ''),
                        'relation': row.get('relation', ''),
                        'target': row.get('target', ''),
                        'context': row.get('context', '') or ''
                    })
                # Détails (definition/exemples/proprietes) si présents dans le graphe
                for name in concept_names:
                    key = name.lower()
                    details[key] = {}
                    def_query = f"""
                    MATCH (d:Definition)-[:LIE_A]->(c:Concept) 
                    WHERE toLower(c.nom) = toLower('{name}')
                    RETURN d.nom as definition LIMIT 1
                    """
                    ex_query = f"""
                    MATCH (c:Concept)-[:A_POUR_EXEMPLE]->(e:Exemple)
                    WHERE toLower(c.nom) = toLower('{name}')
                    RETURN e.nom as exemple LIMIT 3
                    """
                    prop_query = f"""
                    MATCH (p:Propriete)-[:LIE_A]->(c:Concept)
                    WHERE toLower(c.nom) = toLower('{name}')
                    RETURN p.nom as propriete LIMIT 5
                    """
                    def_res = neo4j_manager.query(def_query)
                    ex_res = neo4j_manager.query(ex_query)
                    prop_res = neo4j_manager.query(prop_query)
                    if def_res:
                        details[key]['definition'] = def_res[0].get('definition')
                    if ex_res:
                        details[key]['examples'] = [r.get('exemple') for r in ex_res if r.get('exemple')]
                    if prop_res:
                        details[key]['properties'] = [r.get('propriete') for r in prop_res if r.get('propriete')]
        except Exception as e:
            logger.warning(f"Enrichissement KG (relations/détails) échoué: {e}")

        formatted_response = format_kg_response(concepts, relations=relations, confidence=0.85, details=details if details else None)
        
        # Combiner la réponse LLM avec le formatage KG
        final_response = f"{response}\n\n{formatted_response}"
        
        return final_response
        
    except Exception as e:
        logger.error(f"Erreur génération réponse avec LLM: {e}")
        return generate_enhanced_fallback_response(user_query, concepts)

def generate_enhanced_fallback_response(user_query, concepts):
    """Réponse de fallback améliorée avec formatage structuré"""
    if not concepts:
        return f"Je n'ai pas trouvé de concepts spécifiques pour '{user_query}' dans le Knowledge Graph. Essayez de reformuler votre question ou consultez la liste des concepts disponibles."
    
    # Préparer les détails des concepts
    details = {}
    for concept in concepts:
        concept_name = concept.get('name', '').lower()
        if concept_name:
            details[concept_name] = {
                'definition': f"Concept extrait du module {concept.get('module', 'N/A')}",
                'examples': [f"Utilisé dans {concept.get('module', 'N/A')}"],
                'properties': [f"Module: {concept.get('module', 'N/A')}"]
            }
    
    # Utiliser le formatage KG amélioré
    return format_kg_response(concepts, confidence=0.75, details=details)

def generate_fallback_response(user_query, concepts):
    """Réponse de fallback si le LLM échoue"""
    concept_names = [c['name'] for c in concepts]
    modules = list(set([c['module'] for c in concepts if c['module']]))
    
    response = f"Basé sur votre question '{user_query}', voici ce que j'ai trouvé dans le Knowledge Graph :\n\n"
    response += f"**Concepts trouvés :** {', '.join(concept_names[:5])}\n"
    if modules:
        response += f"**Modules concernés :** {', '.join(modules)}\n\n"
    
    if len(concepts) > 5:
        response += f"... et {len(concepts) - 5} autres concepts.\n\n"
    
    response += "Ces concepts sont extraits de votre base de connaissances et peuvent vous aider à approfondir le sujet."
    
    return response

def initialize_kg():
    """Initialiser le Knowledge Graph"""
    global neo4j_manager
    
    try:
        # Connexion Neo4j
        neo4j_manager = Neo4jManager()
        logger.info("✅ Knowledge Graph connecté à Neo4j")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation KG: {e}")
        neo4j_manager = None

if __name__ == '__main__':
    print("🚀 Démarrage de l'API Knowledge Graph...")
    
    # Initialiser le KG
    initialize_kg()
    
    # Démarrer le serveur
    print("🌐 API KG démarrée sur http://localhost:8001")
    print("📚 Endpoints disponibles:")
    print("   GET  /health       - État de l'API KG")
    print("   POST /kg/query     - Requête KG")
    print("   GET  /kg/concepts  - Liste des concepts")
    print("   POST /kg/search    - Recherche de concepts")
    print("\n⏹️  Appuyez sur Ctrl+C pour arrêter")
    
    app.run(host='0.0.0.0', port=8001, debug=True)
