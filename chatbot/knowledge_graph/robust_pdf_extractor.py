#!/usr/bin/env python3
"""
Extracteur PDF robuste pour gérer différents types de PDF
"""

import os
import re
from pathlib import Path
from typing import Optional, List

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️ PyPDF2 non disponible")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF non disponible")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber non disponible")

class RobustPDFExtractor:
    """Extracteur PDF robuste avec plusieurs méthodes de fallback"""
    
    def __init__(self):
        self.extractors = []
        
        # Ajouter les extracteurs disponibles par ordre de préférence
        if PYMUPDF_AVAILABLE:
            self.extractors.append(self._extract_with_pymupdf)
            print("✅ PyMuPDF disponible")
        
        if PDFPLUMBER_AVAILABLE:
            self.extractors.append(self._extract_with_pdfplumber)
            print("✅ pdfplumber disponible")
        
        if PYPDF2_AVAILABLE:
            self.extractors.append(self._extract_with_pypdf2)
            print("✅ PyPDF2 disponible")
        
        if not self.extractors:
            print("❌ Aucun extracteur PDF disponible")
            print("💡 Installez au moins un de ces packages:")
            print("   pip install PyMuPDF pdfplumber PyPDF2")
    
    def extract_text(self, pdf_path: Path) -> str:
        """Extrait le texte d'un PDF en essayant différentes méthodes"""
        if not self.extractors:
            return ""
        
        for extractor in self.extractors:
            try:
                text = extractor(pdf_path)
                if text and text.strip():
                    # Nettoyer le texte
                    cleaned_text = self._clean_text(text)
                    if cleaned_text:
                        print(f"✅ Texte extrait de {pdf_path.name} avec {extractor.__name__}")
                        return cleaned_text
            except Exception as e:
                print(f"⚠️ {extractor.__name__} échoué pour {pdf_path.name}: {e}")
                continue
        
        print(f"❌ Aucune méthode n'a réussi pour {pdf_path.name}")
        return ""
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extraction avec PyMuPDF (plus robuste)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"PyMuPDF error: {e}")
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extraction avec pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text
        except Exception as e:
            raise Exception(f"pdfplumber error: {e}")
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extraction avec PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"⚠️ Erreur page PyPDF2: {e}")
                        continue
                return text
        except Exception as e:
            raise Exception(f"PyPDF2 error: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte extrait"""
        if not text:
            return ""
        
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normaliser les espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les lignes vides multiples
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Nettoyer les caractères spéciaux
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\'\"]', '', text)
        
        return text.strip()
    
    def get_extractor_info(self) -> dict:
        """Retourne les informations sur les extracteurs disponibles"""
        return {
            "pymupdf": PYMUPDF_AVAILABLE,
            "pdfplumber": PDFPLUMBER_AVAILABLE,
            "pypdf2": PYPDF2_AVAILABLE,
            "total_available": len(self.extractors)
        }

def test_extractor():
    """Test de l'extracteur"""
    extractor = RobustPDFExtractor()
    
    print("🔍 Test de l'extracteur PDF robuste")
    print("=" * 40)
    
    info = extractor.get_extractor_info()
    print(f"Extracteurs disponibles: {info['total_available']}")
    print(f"PyMuPDF: {'✅' if info['pymupdf'] else '❌'}")
    print(f"pdfplumber: {'✅' if info['pdfplumber'] else '❌'}")
    print(f"PyPDF2: {'✅' if info['pypdf2'] else '❌'}")
    
    # Tester avec un PDF si disponible
    docs_path = Path("../docs")
    if docs_path.exists():
        # Chercher récursivement dans tous les sous-dossiers
        pdf_files = list(docs_path.rglob("*.pdf"))
        if pdf_files:
            test_file = pdf_files[0]
            print(f"\n🧪 Test avec {test_file.name}")
            print(f"📁 Chemin complet: {test_file}")
            
            text = extractor.extract_text(test_file)
            if text:
                print(f"✅ Texte extrait: {len(text)} caractères")
                print(f"📝 Aperçu: {text[:200]}...")
            else:
                print("❌ Aucun texte extrait")
        else:
            print("⚠️ Aucun PDF trouvé dans ../docs et ses sous-dossiers")
            print("💡 Vérifiez que les PDF sont bien dans ../docs/BIGDATA/ ou ../docs/SMI/")
    else:
        print("⚠️ Répertoire ../docs non trouvé")
        print("💡 Vérifiez que vous êtes dans le bon répertoire")
        print(f"💡 Répertoire actuel: {Path.cwd()}")
        print(f"💡 Chemin ../docs: {Path('../docs').absolute()}")

if __name__ == "__main__":
    test_extractor()
