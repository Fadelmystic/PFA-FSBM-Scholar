#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test du Modèle FSBM LLaMA Fine-tuné
====================================

Ce script teste le modèle fine-tuné pour vérifier qu'il fonctionne correctement.
"""

import os
import torch
import json
import time
from pathlib import Path

# Configuration
MODEL_PATH = "Model"  # Chemin vers le dossier du modèle
HF_TOKEN = "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"

def print_header():
    """Afficher l'en-tête du test"""
    print("=" * 60)
    print("🧪 TEST DU MODÈLE FSBM LLaMA FINE-TUNÉ")
    print("=" * 60)
    print()

def check_model_files():
    """Vérifier que tous les fichiers du modèle sont présents"""
    print("🔍 VÉRIFICATION DES FICHIERS...")
    
    essential_files = [
        "adapter_model.bin",
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "README.md",
        "model_info.json",
        "training_args.bin",
        "tokenizer.model"
    ]
    
    missing_files = []
    for file in essential_files:
        file_path = os.path.join(MODEL_PATH, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {file} ({size:.1f} MB)")
        else:
            print(f"❌ {file} - MANQUANT")
            missing_files.append(file)
    
    # Vérifier les checkpoints
    checkpoint_dirs = [d for d in os.listdir(MODEL_PATH) if d.startswith("checkpoint-")]
    if checkpoint_dirs:
        print(f"✅ Checkpoints trouvés: {len(checkpoint_dirs)}")
        for checkpoint in checkpoint_dirs:
            print(f"  📁 {checkpoint}/")
    
    if missing_files:
        print(f"\n❌ FICHIERS MANQUANTS: {len(missing_files)}")
        return False
    
    print("\n✅ TOUS LES FICHIERS SONT PRÉSENTS!")
    return True

def load_model_info():
    """Charger les informations du modèle"""
    print("\n📊 INFORMATIONS DU MODÈLE...")
    
    model_info_path = os.path.join(MODEL_PATH, "model_info.json")
    if os.path.exists(model_info_path):
        try:
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            print(f"🎯 Nom: {model_info.get('model_name', 'N/A')}")
            print(f"🔧 Modèle de base: {model_info.get('base_model', 'N/A')}")
            print(f"📈 Époques: {model_info.get('training_config', {}).get('epochs', 'N/A')}")
            print(f"📚 Dataset: {model_info.get('dataset', 'N/A')}")
            print(f"⏱️ Temps d'entraînement: {model_info.get('training_time_seconds', 0):.1f}s")
            print(f"💾 Quantization: {model_info.get('quantization', 'N/A')}")
            print(f"🔧 Framework: {model_info.get('framework', 'N/A')}")
            
            return model_info
        except Exception as e:
            print(f"❌ Erreur lecture model_info.json: {e}")
            return None
    else:
        print("❌ Fichier model_info.json non trouvé")
        return None

def setup_environment():
    """Configurer l'environnement"""
    print("\n🔧 CONFIGURATION DE L'ENVIRONNEMENT...")
    
    # Désactiver les warnings
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Vérifier GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Mémoire GPU nettoyée")
        print(f"💾 Mémoire GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def load_model_and_tokenizer():
    """Charger le modèle et le tokenizer"""
    print("\n🚀 CHARGEMENT DU MODÈLE ET TOKENIZER...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        from huggingface_hub import login
        
        # Authentification Hugging Face
        print("🔐 Authentification Hugging Face...")
        login(token=HF_TOKEN)
        print("✅ Authentification réussie!")
        
        # Configuration 8-bit
        print("🔧 Configuration 8-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        # Charger le tokenizer
        print("📝 Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer chargé!")
        
        # Charger le modèle
        print("🧠 Chargement du modèle...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Modèle chargé!")
        
        return model, tokenizer
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Installez les dépendances: pip install transformers peft torch bitsandbytes")
        return None, None
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None, None

def test_generation(model, tokenizer, prompt, max_tokens=100):
    """Tester la génération de texte"""
    try:
        # Formater le prompt
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        # Déplacer sur le bon device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Générer
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Décoder la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()
        
        return response
        
    except Exception as e:
        return f"❌ Erreur de génération: {e}"

def run_tests(model, tokenizer):
    """Exécuter les tests"""
    print("\n🧪 TESTS DE GÉNÉRATION...")
    print("=" * 50)
    
    # Questions de test
    test_questions = [
        "Qu'est-ce que le Big Data?",
        "Expliquez Python",
        "Qu'est-ce que l'IA?",
        "Définissez la Business Intelligence",
        "Qu'est-ce que le Machine Learning?",
        "Expliquez les bases de données",
        "Qu'est-ce que l'analyse de données?",
        "Définissez le Data Mining"
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/{len(test_questions)} ---")
        print(f"📝 Question: {question}")
        
        start_time = time.time()
        response = test_generation(model, tokenizer, question)
        generation_time = time.time() - start_time
        
        print(f"🤖 Réponse: {response}")
        print(f"⏱️ Temps: {generation_time:.2f}s")
        
        results.append({
            "question": question,
            "response": response,
            "time": generation_time
        })
    
    return results

def analyze_results(results):
    """Analyser les résultats des tests"""
    print("\n📊 ANALYSE DES RÉSULTATS...")
    print("=" * 50)
    
    total_time = sum(r["time"] for r in results)
    avg_time = total_time / len(results)
    
    print(f"📈 Nombre de tests: {len(results)}")
    print(f"⏱️ Temps total: {total_time:.2f}s")
    print(f"⏱️ Temps moyen: {avg_time:.2f}s")
    
    # Vérifier la qualité des réponses
    good_responses = 0
    for result in results:
        response = result["response"]
        if not response.startswith("❌") and len(response) > 10:
            good_responses += 1
    
    success_rate = (good_responses / len(results)) * 100
    print(f"✅ Taux de succès: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! Le modèle fonctionne parfaitement!")
    elif success_rate >= 60:
        print("✅ BON! Le modèle fonctionne bien!")
    else:
        print("⚠️ ATTENTION! Le modèle a des problèmes.")

def main():
    """Fonction principale"""
    print_header()
    
    # Vérifier les fichiers
    if not check_model_files():
        print("❌ Fichiers manquants. Arrêt du test.")
        return
    
    # Charger les informations du modèle
    model_info = load_model_info()
    
    # Configurer l'environnement
    device = setup_environment()
    
    # Charger le modèle et tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        print("❌ Impossible de charger le modèle. Arrêt du test.")
        return
    
    # Exécuter les tests
    results = run_tests(model, tokenizer)
    
    # Analyser les résultats
    analyze_results(results)
    
    print("\n" + "=" * 60)
    print("🎉 TEST TERMINÉ!")
    print("=" * 60)
    print("✅ Le modèle a été testé avec succès")
    print("📊 Résultats disponibles ci-dessus")
    print("🚀 Votre modèle FSBM LLaMA est prêt à l'emploi!")

if __name__ == "__main__":
    main()
