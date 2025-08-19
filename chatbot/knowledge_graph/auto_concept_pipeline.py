#!/usr/bin/env python3
"""
Pipeline Automatisé d'Extraction de Concepts pour FSBM Scholar Assistant
Surveille automatiquement les nouveaux documents et relance l'extraction
"""

import os
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
import schedule
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dynamic_concept_extractor import DynamicConceptExtractor
from enhanced_kg_chatbot import EnhancedKGQueryEngine
from neo4j_manager import Neo4jManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentChangeHandler(FileSystemEventHandler):
    """Gestionnaire des changements de documents"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.last_processed = {}
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logger.info(f"📄 Nouveau document détecté: {event.src_path}")
            self.pipeline.schedule_extraction()
            
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logger.info(f"📝 Document modifié: {event.src_path}")
            self.pipeline.schedule_extraction()
            
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logger.info(f"🗑️ Document supprimé: {event.src_path}")
            self.pipeline.schedule_extraction()

class AutoConceptPipeline:
    """Pipeline automatisé d'extraction de concepts"""
    
    def __init__(self, docs_path: str = "../docs", config_file: str = "pipeline_config.json"):
        self.docs_path = Path(docs_path)
        self.config_file = config_file
        
        # Utiliser l'extracteur robuste si disponible
        try:
            from robust_pdf_extractor import RobustPDFExtractor
            self.extractor = DynamicConceptExtractor()
            self.robust_extractor = RobustPDFExtractor()
            logger.info("✅ Extracteur PDF robuste disponible")
        except ImportError:
            self.extractor = DynamicConceptExtractor()
            self.robust_extractor = None
            logger.info("⚠️ Extracteur PDF robuste non disponible, utilisation de l'extracteur standard")
        
        self.neo4j_manager = None
        self.observer = None
        self.change_handler = None
        
        # Configuration par défaut
        self.config = self.load_config()
        
        # État du pipeline
        self.last_extraction = None
        self.extraction_scheduled = False
        self.processing = False
        
        # Initialiser la base de données si configurée
        if self.config.get("use_neo4j", False):
            try:
                neo4j_config = self.config.get("neo4j_config", {})
                self.neo4j_manager = Neo4jManager(
                    uri=neo4j_config.get("uri"),
                    user=neo4j_config.get("user"),
                    password=neo4j_config.get("password")
                )
                logger.info("✅ Connexion Neo4j établie")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de se connecter à Neo4j: {e}")
                logger.info("💡 Le pipeline continuera sans Neo4j")
                self.neo4j_manager = None
        
        logger.info("🚀 Pipeline d'extraction automatique initialisé")
        
    def load_config(self) -> Dict:
        """Charge la configuration du pipeline"""
        default_config = {
            "auto_extract": True,
            "extraction_interval": 3600,  # 1 heure
            "watch_directory": True,
            "use_neo4j": True,
            "backup_results": True,
            "max_concepts_per_module": 100,
            "min_concept_frequency": 2,
            "extraction_timeout": 300,  # 5 minutes
            "notify_changes": True
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    logger.info("✅ Configuration chargée")
            except Exception as e:
                logger.warning(f"⚠️ Erreur de chargement de la config: {e}")
        
        # Sauvegarder la configuration par défaut
        self.save_config(default_config)
        return default_config
    
    def save_config(self, config: Dict):
        """Sauvegarde la configuration"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Erreur de sauvegarde de la config: {e}")
    
    def get_documents_hash(self) -> str:
        """Calcule le hash de tous les documents pour détecter les changements"""
        all_hashes = []
        
        for pdf_file in self.docs_path.rglob("*.pdf"):
            try:
                stat = pdf_file.stat()
                file_hash = f"{pdf_file.name}:{stat.st_mtime}:{stat.st_size}"
                all_hashes.append(file_hash)
            except Exception as e:
                logger.warning(f"⚠️ Erreur avec {pdf_file}: {e}")
        
        # Trier pour avoir un hash cohérent
        all_hashes.sort()
        combined = "|".join(all_hashes)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def has_documents_changed(self) -> bool:
        """Vérifie si les documents ont changé depuis la dernière extraction"""
        current_hash = self.get_documents_hash()
        
        if not self.last_extraction:
            return True
            
        last_hash = self.last_extraction.get("documents_hash")
        return current_hash != last_hash
    
    def extract_concepts(self) -> Dict:
        """Lance l'extraction de concepts"""
        if self.processing:
            logger.info("⏳ Extraction déjà en cours, ignorée")
            return None
            
        self.processing = True
        start_time = time.time()
        
        try:
            logger.info("🔍 Lancement de l'extraction de concepts...")
            
            # Extraire tous les concepts
            results = self.extractor.extract_all_concepts_from_docs(self.docs_path)
            
            # Ajouter les métadonnées
            results["extraction_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "documents_hash": self.get_documents_hash(),
                "processing_time": time.time() - start_time,
                "total_documents": len(results.get("documents", [])),
                "total_concepts": sum(len(concepts) for concepts in results.get("concepts", {}).values())
            }
            
            # Sauvegarder les résultats
            self.save_extraction_results(results)
            
            # Mettre à jour la base de données si configurée
            if self.neo4j_manager and self.config.get("use_neo4j"):
                self.update_neo4j_database(results)
            
            # Sauvegarder l'état
            self.last_extraction = results["extraction_metadata"]
            
            logger.info(f"✅ Extraction terminée en {time.time() - start_time:.2f}s")
            logger.info(f"📊 {results['extraction_metadata']['total_concepts']} concepts extraits")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction: {e}")
            return None
        finally:
            self.processing = False
    
    def save_extraction_results(self, results: Dict):
        """Sauvegarde les résultats d'extraction"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarde principale
        main_file = f"extraction_results_{timestamp}.json"
        try:
            with open(main_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Résultats sauvegardés: {main_file}")
        except Exception as e:
            logger.error(f"❌ Erreur de sauvegarde: {e}")
        
        # Sauvegarde de sauvegarde si configurée
        if self.config.get("backup_results", True):
            backup_file = f"backup/extraction_results_{timestamp}.json"
            try:
                os.makedirs("backup", exist_ok=True)
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ Sauvegarde créée: {backup_file}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur de sauvegarde: {e}")
        
        # Mettre à jour le fichier principal
        try:
            with open("latest_extraction_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Erreur de mise à jour du fichier principal: {e}")
    
    def update_neo4j_database(self, results: Dict):
        """Met à jour la base de données Neo4j avec les nouveaux concepts"""
        if not self.neo4j_manager:
            return
            
        try:
            logger.info("🔄 Mise à jour de la base Neo4j...")
            
            # Créer les nœuds de concepts
            for module, concepts in results.get("concepts", {}).items():
                for concept in concepts:
                    self.neo4j_manager.create_concept_node(concept, module)
            
            # Créer les relations
            for relation in results.get("relations", []):
                self.neo4j_manager.create_relation(
                    relation["source"], 
                    relation["target"], 
                    relation["type"]
                )
            
            logger.info("✅ Base Neo4j mise à jour")
            
        except Exception as e:
            logger.error(f"❌ Erreur de mise à jour Neo4j: {e}")
    
    def schedule_extraction(self):
        """Planifie une extraction si pas déjà planifiée"""
        if not self.extraction_scheduled:
            self.extraction_scheduled = True
            logger.info("⏰ Extraction planifiée dans 30 secondes...")
            
            # Planifier l'extraction dans 30 secondes
            schedule.every(30).seconds.do(self.run_scheduled_extraction).tag('extraction')
    
    def run_scheduled_extraction(self):
        """Exécute l'extraction planifiée"""
        if self.extraction_scheduled:
            self.extraction_scheduled = False
            schedule.clear('extraction')
            
            if self.has_documents_changed():
                logger.info("🔄 Changements détectés, lancement de l'extraction...")
                self.extract_concepts()
            else:
                logger.info("✅ Aucun changement détecté")
    
    def start_watching(self):
        """Démarre la surveillance du répertoire"""
        if not self.config.get("watch_directory", True):
            return
            
        try:
            self.observer = Observer()
            self.change_handler = DocumentChangeHandler(self)
            
            self.observer.schedule(
                self.change_handler, 
                str(self.docs_path), 
                recursive=True
            )
            
            self.observer.start()
            logger.info(f"👀 Surveillance démarrée sur {self.docs_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur de démarrage de la surveillance: {e}")
    
    def start_scheduled_extraction(self):
        """Démarre l'extraction planifiée"""
        if not self.config.get("auto_extract", True):
            return
            
        interval = self.config.get("extraction_interval", 3600)
        schedule.every(interval).seconds.do(self.run_scheduled_extraction)
        logger.info(f"⏰ Extraction planifiée toutes les {interval} secondes")
    
    def run(self):
        """Lance le pipeline complet"""
        logger.info("🚀 Démarrage du pipeline automatique...")
        
        # Première extraction si nécessaire
        if self.has_documents_changed():
            logger.info("🔄 Première extraction...")
            self.extract_concepts()
        
        # Démarrer la surveillance
        self.start_watching()
        
        # Démarrer l'extraction planifiée
        self.start_scheduled_extraction()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt du pipeline...")
            self.stop()
    
    def stop(self):
        """Arrête le pipeline"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        logger.info("✅ Pipeline arrêté")
    
    def get_status(self) -> Dict:
        """Retourne le statut du pipeline"""
        return {
            "status": "running" if not self.processing else "processing",
            "last_extraction": self.last_extraction,
            "extraction_scheduled": self.extraction_scheduled,
            "documents_hash": self.get_documents_hash(),
            "config": self.config
        }

def main():
    """Fonction principale"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline automatique d'extraction de concepts")
    parser.add_argument("--docs", default="../docs", help="Chemin vers les documents")
    parser.add_argument("--config", default="pipeline_config.json", help="Fichier de configuration")
    parser.add_argument("--extract-now", action="store_true", help="Lancer une extraction immédiatement")
    parser.add_argument("--status", action="store_true", help="Afficher le statut")
    
    args = parser.parse_args()
    
    pipeline = AutoConceptPipeline(args.docs, args.config)
    
    if args.status:
        status = pipeline.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return
    
    if args.extract_now:
        pipeline.extract_concepts()
        return
    
    # Lancer le pipeline complet
    try:
        pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()

if __name__ == "__main__":
    main()
