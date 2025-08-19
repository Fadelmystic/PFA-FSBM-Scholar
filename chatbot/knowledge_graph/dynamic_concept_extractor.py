#!/usr/bin/env python3
"""
Extracteur dynamique de concepts pour FSBM Scholar Assistant
Analyse tous les documents PDF pour détecter automatiquement les concepts
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import PyPDF2
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DynamicConceptExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('french') + stopwords.words('english'))
        self.technical_indicators = [
            'définition', 'concept', 'principe', 'méthode', 'technique', 'algorithme',
            'framework', 'library', 'tool', 'technology', 'protocol', 'standard',
            'pattern', 'architecture', 'model', 'system', 'platform', 'service',
            'api', 'sdk', 'ide', 'compiler', 'interpreter', 'runtime', 'virtual machine'
        ]
        
        # Patterns pour détecter les concepts techniques
        self.concept_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Mots avec majuscules
            r'\b[A-Z]{2,}\b',  # Acronymes
            r'\b[a-z]+(?:\.[a-z]+)+\b',  # Noms de packages
            r'\b[A-Z][a-z]+(?:\d+)?\b',  # Noms de technologies
        ]
        
        # Mots-clés techniques spécifiques
        self.tech_keywords = {
            'java': ['java', 'jdk', 'jre', 'jvm', 'bytecode', 'servlet', 'jsp', 'jstl', 'jdbc', 'jpa', 'ejb', 'jee'],
            'web': ['html', 'css', 'javascript', 'php', 'python', 'ruby', 'node.js', 'react', 'angular', 'vue'],
            'database': ['sql', 'mysql', 'postgresql', 'oracle', 'mongodb', 'redis', 'cassandra', 'neo4j'],
            'bigdata': ['hadoop', 'spark', 'hive', 'hbase', 'kafka', 'storm', 'flink', 'airflow'],
            'ml': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'pandas', 'numpy', 'matplotlib'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab'],
            'security': ['ssl', 'tls', 'oauth', 'jwt', 'encryption', 'hashing', 'blockchain', 'cryptography'],
            'devops': ['ci/cd', 'continuous integration', 'continuous deployment', 'microservices', 'api gateway']
        }

    def extract_all_concepts_from_docs(self, docs_path: str = "../docs") -> Dict:
        """Extrait tous les concepts de tous les documents PDF"""
        print("🔍 Extraction dynamique de tous les concepts...")
        
        all_texts = []
        module_concepts = defaultdict(set)
        document_info = []
        
        docs_dir = Path(docs_path)
        if not docs_dir.exists():
            print(f"❌ Dossier {docs_path} non trouvé")
            return {"concepts": {}, "modules": {}, "documents": []}
        
        # Parcourir tous les PDF
        for pdf_file in docs_dir.rglob("*.pdf"):
            try:
                print(f"📖 Analyse: {pdf_file.name}")
                text = self._extract_pdf_text(pdf_file)
                if text:
                    # Extraire le module depuis le chemin
                    module = self._extract_module_from_path(pdf_file)
                    
                    # Analyser le texte pour les concepts
                    concepts = self._extract_concepts_from_text(text)
                    
                    # Organiser par module
                    for concept in concepts:
                        module_concepts[module].add(concept)
                    
                    all_texts.append(text)
                    document_info.append({
                        "file": pdf_file.name,
                        "module": module,
                        "concepts_count": len(concepts),
                        "concepts": list(concepts)
                    })
                    
            except Exception as e:
                print(f"⚠️ Erreur avec {pdf_file.name}: {e}")
        
        # Analyser tous les textes pour détecter les concepts globaux
        global_concepts = self._analyze_global_concepts(all_texts)
        
        # Créer la hiérarchie des concepts
        concept_hierarchy = self._create_concept_hierarchy(global_concepts, module_concepts)
        
        return {
            "concepts": {k: list(v) for k, v in module_concepts.items()},
            "global_concepts": global_concepts,
            "hierarchy": concept_hierarchy,
            "documents": document_info
        }

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extrait le texte d'un PDF avec gestion d'erreurs robuste"""
        try:
            # Essayer d'abord avec PyPDF2
            with open(pdf_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as e:
                            print(f"⚠️ Erreur page PDF: {e}")
                            continue
                    
                    if text.strip():
                        return text
                    else:
                        print(f"⚠️ Aucun texte extrait de {pdf_path.name}")
                        return ""
                        
                except Exception as e:
                    print(f"⚠️ Erreur PyPDF2 avec {pdf_path.name}: {e}")
                    return ""
                    
        except Exception as e:
            print(f"❌ Erreur lecture fichier {pdf_path.name}: {e}")
            return ""

    def _extract_module_from_path(self, pdf_path: Path) -> str:
        """Extrait le nom du module depuis le chemin du fichier"""
        parts = pdf_path.parts
        for part in parts:
            if part.upper() in ['JAVA', 'JEE', 'BIGDATA', 'ML', 'SECURITE', 'BDD', 'WEB', 'DEVOPS']:
                return part.upper()
            elif 'seance' in part.lower() or 'chapitre' in part.lower():
                # Remonter au dossier parent
                idx = parts.index(part)
                if idx > 0:
                    return parts[idx-1].upper()
        return "AUTRE"

    def _extract_concepts_from_text(self, text: str) -> Set[str]:
        """Extrait les concepts d'un texte"""
        concepts = set()
        
        # Tokenisation
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            # Détecter les concepts par patterns
            for pattern in self.concept_patterns:
                matches = re.findall(pattern, sentence)
                for match in matches:
                    if self._is_valid_concept(match):
                        concepts.add(match.lower())
            
            # Détecter les concepts techniques
            for category, keywords in self.tech_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in sentence.lower():
                        concepts.add(keyword.lower())
            
            # Détecter les concepts autour des mots-clés techniques
            for indicator in self.technical_indicators:
                if indicator in sentence.lower():
                    # Extraire les mots autour de l'indicateur
                    words = word_tokenize(sentence)
                    for i, word in enumerate(words):
                        if indicator in word.lower():
                            # Prendre les mots avant et après
                            context_words = words[max(0, i-2):i+3]
                            for ctx_word in context_words:
                                if self._is_valid_concept(ctx_word):
                                    concepts.add(ctx_word.lower())
        
        return concepts

    def _is_valid_concept(self, word: str) -> bool:
        """Vérifie si un mot est un concept valide"""
        if len(word) < 3:
            return False
        
        if word.lower() in self.stop_words:
            return False
        
        # Mots trop communs
        common_words = {'cours', 'module', 'chapitre', 'partie', 'section', 'page', 'document'}
        if word.lower() in common_words:
            return False
        
        return True

    def _analyze_global_concepts(self, texts: List[str]) -> Dict:
        """Analyse tous les textes pour détecter les concepts globaux"""
        all_words = []
        
        for text in texts:
            words = word_tokenize(text.lower())
            all_words.extend([w for w in words if self._is_valid_concept(w)])
        
        # Compter les occurrences
        word_counts = Counter(all_words)
        
        # Filtrer les mots fréquents (concepts importants)
        frequent_concepts = {word: count for word, count in word_counts.items() 
                           if count >= 3 and len(word) >= 4}
        
        return dict(frequent_concepts)

    def _create_concept_hierarchy(self, global_concepts: Dict, module_concepts: Dict) -> Dict:
        """Crée une hiérarchie des concepts"""
        hierarchy = {
            "technologies": {},
            "concepts": {},
            "modules": {}
        }
        
        # Organiser par catégories
        for concept, count in global_concepts.items():
            category = self._categorize_concept(concept)
            if category not in hierarchy["concepts"]:
                hierarchy["concepts"][category] = []
            hierarchy["concepts"][category].append({
                "name": concept,
                "frequency": count,
                "modules": [mod for mod, concepts in module_concepts.items() if concept in concepts]
            })
        
        # Organiser par modules
        for module, concepts in module_concepts.items():
            hierarchy["modules"][module] = {
                "concepts": list(concepts),
                "count": len(concepts)
            }
        
        return hierarchy

    def _categorize_concept(self, concept: str) -> str:
        """Catégorise un concept"""
        concept_lower = concept.lower()
        
        # Vérifier les catégories techniques
        for category, keywords in self.tech_keywords.items():
            if any(keyword in concept_lower for keyword in keywords):
                return category
        
        # Catégories par patterns
        if re.match(r'^[A-Z]{2,}$', concept):
            return "acronyms"
        elif '.' in concept:
            return "packages"
        elif re.match(r'^[A-Z][a-z]+(?:\d+)?$', concept):
            return "technologies"
        else:
            return "general"

    def generate_concept_keywords(self, hierarchy: Dict) -> Dict[str, List[str]]:
        """Génère automatiquement les mots-clés pour le chatbot"""
        concept_keywords = {}
        
        # Ajouter les concepts par catégorie
        for category, concepts in hierarchy["concepts"].items():
            if category not in concept_keywords:
                concept_keywords[category] = []
            
            for concept_info in concepts:
                concept_name = concept_info["name"]
                concept_keywords[category].append(concept_name)
                
                # Ajouter des variations
                if ' ' in concept_name:
                    # Pour les concepts multi-mots, ajouter des variations
                    words = concept_name.split()
                    concept_keywords[category].extend(words)
        
        # Ajouter les concepts par module
        for module, module_info in hierarchy["modules"].items():
            module_lower = module.lower()
            if module_lower not in concept_keywords:
                concept_keywords[module_lower] = []
            
            for concept in module_info["concepts"]:
                concept_keywords[module_lower].append(concept)
        
        return concept_keywords

    def save_extraction_results(self, results: Dict, output_file: str = "dynamic_concepts.json"):
        """Sauvegarde les résultats d'extraction"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Résultats sauvegardés dans {output_file}")
        
        # Générer les mots-clés pour le chatbot
        concept_keywords = self.generate_concept_keywords(results["hierarchy"])
        
        # Sauvegarder les mots-clés
        keywords_file = "generated_concept_keywords.json"
        with open(keywords_file, 'w', encoding='utf-8') as f:
            json.dump(concept_keywords, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Mots-clés générés sauvegardés dans {keywords_file}")
        
        # Afficher les statistiques
        print(f"\n📊 Statistiques d'extraction:")
        print(f"   • Modules analysés: {len(results['concepts'])}")
        print(f"   • Concepts globaux: {len(results['global_concepts'])}")
        print(f"   • Documents traités: {len(results['documents'])}")
        
        # Afficher les concepts les plus fréquents
        print(f"\n🔝 Concepts les plus fréquents:")
        sorted_concepts = sorted(results['global_concepts'].items(), key=lambda x: x[1], reverse=True)
        for concept, count in sorted_concepts[:10]:
            print(f"   • {concept}: {count} occurrences")
        
        return results

if __name__ == "__main__":
    extractor = DynamicConceptExtractor()
    results = extractor.extract_all_concepts_from_docs()
    extractor.save_extraction_results(results)
    
    print("\n🎯 Prochaines étapes:")
    print("1. Vérifiez les concepts extraits dans dynamic_concepts.json")
    print("2. Utilisez generated_concept_keywords.json pour mettre à jour le chatbot")
    print("3. Relancez le chatbot avec les nouveaux concepts")
