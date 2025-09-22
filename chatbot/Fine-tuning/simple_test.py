#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Simple - Modèle FSBM LLaMA Fine-tuné
==========================================

Test simple du tokenizer et vérification des fichiers.
"""

import os
import json

# Configuration
MODEL_PATH = "Model"

def test_files():
    """Tester la présence des fichiers"""
    print("🔍 VÉRIFICATION DES FICHIERS...")
    print("=" * 50)
    
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
    
    all_files_present = True
    for file in essential_files:
        file_path = os.path.join(MODEL_PATH, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ {file} ({size:.1f} MB)")
        else:
            print(f"❌ {file} - MANQUANT")
            all_files_present = False
    
    return all_files_present

def test_tokenizer():
    """Tester le tokenizer"""
    print("\n📝 TEST DU TOKENIZER...")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer
        
        print("🔧 Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Tokenizer chargé!")
        
        # Test de tokenization
        test_text = "Qu'est-ce que le Big Data?"
        formatted_prompt = f"<s>[INST] {test_text} [/INST]"
        
        print(f"📝 Test de tokenization: '{test_text}'")
        tokens = tokenizer.encode(formatted_prompt, return_tensors="pt")
        print(f"✅ Tokens générés: {tokens.shape}")
        
        # Test de décodage
        decoded = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print(f"✅ Décodage réussi: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur tokenizer: {e}")
        return False

def show_model_info():
    """Afficher les informations du modèle"""
    print("\n📊 INFORMATIONS DU MODÈLE...")
    print("=" * 50)
    
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
            
        except Exception as e:
            print(f"❌ Erreur lecture model_info.json: {e}")
    else:
        print("❌ Fichier model_info.json non trouvé")

def main():
    """Fonction principale"""
    print("🚀 TEST SIMPLE - MODÈLE FSBM LLaMA")
    print("=" * 60)
    
    # Test des fichiers
    files_ok = test_files()
    
    if not files_ok:
        print("\n❌ FICHIERS MANQUANTS!")
        print("💡 Vérifiez que le modèle est correctement téléchargé")
        return
    
    # Test du tokenizer
    tokenizer_ok = test_tokenizer()
    
    # Afficher les informations
    show_model_info()
    
    print("\n" + "=" * 60)
    if files_ok and tokenizer_ok:
        print("🎉 TEST RÉUSSI!")
        print("✅ Tous les fichiers sont présents")
        print("✅ Le tokenizer fonctionne")
        print("💡 Le modèle est prêt pour utilisation")
        print("\n💡 Prochaines étapes:")
        print("1. python quick_test_cpu.py - Test rapide sur CPU")
        print("2. python interactive_chat_cpu.py - Chat interactif sur CPU")
    else:
        print("❌ PROBLÈMES DÉTECTÉS!")
        print("💡 Vérifiez les erreurs ci-dessus")

if __name__ == "__main__":
    main()
