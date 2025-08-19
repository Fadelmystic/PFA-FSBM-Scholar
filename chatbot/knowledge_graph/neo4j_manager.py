from typing import List, Dict, Optional
from neo4j import GraphDatabase
import os

# Configuration par défaut Neo4j
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "password"

class Neo4jManager:
    def __init__(self, uri: str = None, user: str = None, password: str = None) -> None:
        self.uri = uri or os.getenv('NEO4J_URI', DEFAULT_NEO4J_URI)
        self.user = user or os.getenv('NEO4J_USER', DEFAULT_NEO4J_USER)
        self.password = password or os.getenv('NEO4J_PASSWORD', DEFAULT_NEO4J_PASSWORD)
        self.driver = None
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test de connexion
            if self.test_connection():
                print("✅ Connexion Neo4j établie")
            else:
                raise Exception("Test de connexion échoué")
        except Exception as e:
            print(f"❌ Erreur de connexion Neo4j: {e}")
            print("💡 Vérifiez que Neo4j est démarré et accessible")
            raise

    def close(self) -> None:
        if self.driver:
            self.driver.close()

    def test_connection(self) -> bool:
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS ok")
                return result.single()["ok"] == 1
        except Exception as e:
            print(f"❌ Test de connexion échoué: {e}")
            return False

    def create_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cours) REQUIRE c.nom IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Personne) REQUIRE p.nom IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (x:Concept) REQUIRE x.nom IS UNIQUE",
        ]
        with self.driver.session() as session:
            for s in statements:
                try:
                    session.run(s)
                    print(f"✅ Contrainte créée: {s}")
                except Exception as e:
                    print(f"⚠️ Contrainte déjà existante ou erreur: {e}")

    def upsert_entity(self, label: str, name: str, properties: Dict) -> None:
        with self.driver.session() as session:
            props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            query = (
                f"MERGE (e:{label} {{nom: $name}}) "
                f"SET e += {{{props_str}}}"
            )
            params = {"name": name, **properties}
            session.run(query, params)

    def upsert_relation(
        self,
        from_label: str,
        from_name: str,
        rel_type: str,
        to_label: str,
        to_name: str,
        properties: Optional[Dict] = None,
    ) -> None:
        with self.driver.session() as session:
            session.run(
                f"""
                MATCH (a:{from_label} {{nom: $from_name}})
                MATCH (b:{to_label} {{nom: $to_name}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $props
                """,
                from_name=from_name,
                to_name=to_name,
                props=properties or {},
            )

    def query(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        with self.driver.session() as session:
            res = session.run(cypher, **(params or {}))
            return [r.data() for r in res]

    # Méthodes manquantes pour le pipeline
    def create_concept_node(self, concept_name: str, module: str) -> None:
        """Crée un nœud de concept"""
        try:
            with self.driver.session() as session:
                query = """
                MERGE (c:Concept {nom: $concept_name})
                SET c.module = $module, c.timestamp = datetime()
                """
                session.run(query, concept_name=concept_name, module=module)
                print(f"✅ Concept créé: {concept_name} (module: {module})")
        except Exception as e:
            print(f"❌ Erreur création concept {concept_name}: {e}")

    def create_relation(self, source: str, target: str, rel_type: str) -> None:
        """Crée une relation entre deux concepts"""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (a:Concept {nom: $source})
                MATCH (b:Concept {nom: $target})
                MERGE (a)-[r:RELATES_TO {type: $rel_type}]->(b)
                SET r.timestamp = datetime()
                """
                session.run(query, source=source, target=target, rel_type=rel_type)
                print(f"✅ Relation créée: {source} -> {target} ({rel_type})")
        except Exception as e:
            print(f"❌ Erreur création relation {source} -> {target}: {e}")

    def delete_concept_node(self, concept_name: str) -> None:
        """Supprime un nœud de concept (pour les tests)"""
        try:
            with self.driver.session() as session:
                query = "MATCH (c:Concept {nom: $concept_name}) DETACH DELETE c"
                session.run(query, concept_name=concept_name)
                print(f"✅ Concept supprimé: {concept_name}")
        except Exception as e:
            print(f"❌ Erreur suppression concept {concept_name}: {e}")

    def get_concepts_by_module(self, module: str) -> List[Dict]:
        """Récupère tous les concepts d'un module"""
        try:
            with self.driver.session() as session:
                query = "MATCH (c:Concept {module: $module}) RETURN c.nom as nom, c.module as module"
                result = session.run(query, module=module)
                return [record.data() for record in result]
        except Exception as e:
            print(f"❌ Erreur récupération concepts module {module}: {e}")
            return []

    def get_all_concepts(self) -> List[Dict]:
        """Récupère tous les concepts"""
        try:
            with self.driver.session() as session:
                query = "MATCH (c:Concept) RETURN c.nom as nom, c.module as module"
                result = session.run(query)
                return [record.data() for record in result]
        except Exception as e:
            print(f"❌ Erreur récupération tous concepts: {e}")
            return []

    def clear_all_data(self) -> None:
        """Supprime toutes les données (pour les tests)"""
        try:
            with self.driver.session() as session:
                query = "MATCH (n) DETACH DELETE n"
                session.run(query)
                print("✅ Toutes les données supprimées")
        except Exception as e:
            print(f"❌ Erreur suppression données: {e}")

    def get_statistics(self) -> Dict:
        """Récupère les statistiques de la base"""
        try:
            with self.driver.session() as session:
                stats = {}
                
                # Nombre de concepts
                result = session.run("MATCH (c:Concept) RETURN count(c) as count")
                stats['concepts_count'] = result.single()['count']
                
                # Nombre de relations
                result = session.run("MATCH ()-[r]-() RETURN count(r) as count")
                stats['relations_count'] = result.single()['count']
                
                # Modules (filtrer les valeurs None)
                result = session.run("MATCH (c:Concept) RETURN DISTINCT c.module as module")
                modules = [record['module'] for record in result if record['module'] is not None]
                stats['modules'] = modules
                
                return stats
        except Exception as e:
            print(f"❌ Erreur récupération statistiques: {e}")
            return {}
