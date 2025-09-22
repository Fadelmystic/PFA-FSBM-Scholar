#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Interactif CPU Corrigé - Modèle FSBM LLaMA Fine-tuné
==========================================================

Version corrigée du chat interactif pour résoudre les problèmes d'attention mask.
"""

import os
import torch
import json
from pathlib import Path

# Configuration
MODEL_PATH = "Model"
HF_TOKEN = "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"

def print_header():
    """Afficher l'en-tête du chat"""
    print("=" * 60)
    print("🎓 FSBM SCHOLAR ASSISTANT - CHAT INTERACTIF (CPU CORRIGÉ)")
    print("=" * 60)
    print("🤖 Assistant IA fine-tuné pour les questions FSBM")
    print("💻 Mode CPU - Version corrigée")
    print("💡 Tapez 'help' pour voir les questions d'exemple")
    print("🚪 Tapez 'quit' pour quitter")
    print("=" * 60)
    print()

def load_model():
    """Charger le modèle fine-tuné sur CPU"""
    print("🔧 Chargement du modèle sur CPU...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import login
        
        # Configuration
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Authentification
        login(token=HF_TOKEN)
        
        # Charger le tokenizer
        print("📝 Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # Configuration du tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # S'assurer que le tokenizer a un token de padding différent
        if tokenizer.pad_token == tokenizer.eos_token:
            # Créer un token de padding unique
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        print("✅ Tokenizer chargé!")
        
        # Charger le modèle sur CPU
        print("🧠 Chargement du modèle (CPU)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cpu",  # Forcer CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32  # Utiliser float32 pour CPU
        )
        
        # Redimensionner l'embedding si nécessaire
        if tokenizer.pad_token != tokenizer.eos_token:
            model.resize_token_embeddings(len(tokenizer))
        
        print("✅ Modèle chargé avec succès sur CPU!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_tokens=150):
    """Générer une réponse avec attention mask corrigé"""
    try:
        # Formater le prompt
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Tokenization avec attention mask
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True
        )
        
        # Forcer sur CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Générer avec attention mask
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=inputs['attention_mask']
            )
        
        # Décoder la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie réponse
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()
        
        return response
        
    except Exception as e:
        return f"❌ Erreur de génération: {e}"

def show_help():
    """Afficher l'aide et les questions d'exemple"""
    print("\n📚 QUESTIONS D'EXEMPLE:")
    print("-" * 40)
    
    sample_questions = [
        "Qu'est-ce que le Big Data?",
        "Expliquez les 3 V du Big Data",
        "Qu'est-ce que la Business Intelligence?",
        "Définissez le Machine Learning",
        "Qu'est-ce que l'analyse de données?",
        "Expliquez les bases de données",
        "Qu'est-ce que le Data Mining?",
        "Définissez l'Intelligence Artificielle",
        "Qu'est-ce qu'Apache Hadoop?",
        "Expliquez Python pour la data science",
        "Qu'est-ce que la validation croisée?",
        "Définissez l'ingénierie des caractéristiques"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"{i:2d}. {question}")
    
    print("\n💡 COMMANDES SPÉCIALES:")
    print("- help : Afficher cette aide")
    print("- quit : Quitter le chat")
    print("- clear : Effacer l'écran")
    print("- info : Informations sur le modèle")

def show_model_info():
    """Afficher les informations du modèle"""
    print("\n📊 INFORMATIONS DU MODÈLE:")
    print("-" * 40)
    
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
            print(f"💻 Mode: CPU (version corrigée)")
            
        except Exception as e:
            print(f"❌ Erreur lecture model_info.json: {e}")
    else:
        print("❌ Fichier model_info.json non trouvé")

def clear_screen():
    """Effacer l'écran"""
    os.system('cls' if os.name == 'nt' else 'clear')
    print_header()

def interactive_chat():
    """Interface de chat interactive corrigée"""
    print_header()
    
    # Charger le modèle
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("❌ Impossible de charger le modèle. Arrêt du chat.")
        return
    
    print("\n🎉 Prêt pour le chat! Posez vos questions FSBM...")
    print("⚠️ Mode CPU - Les réponses peuvent être plus lentes")
    print("🔧 Version corrigée - Attention mask géré")
    print()
    
    # Statistiques
    question_count = 0
    total_time = 0
    
    while True:
        try:
            # Obtenir l'entrée utilisateur
            user_input = input("\n🤔 Vous: ").strip()
            
            # Commandes spéciales
            if user_input.lower() == 'quit':
                print("\n👋 Au revoir! Merci d'avoir testé le modèle FSBM!")
                if question_count > 0:
                    avg_time = total_time / question_count
                    print(f"📊 Statistiques: {question_count} questions, temps moyen: {avg_time:.2f}s")
                break
                
            elif user_input.lower() == 'help':
                show_help()
                continue
                
            elif user_input.lower() == 'info':
                show_model_info()
                continue
                
            elif user_input.lower() == 'clear':
                clear_screen()
                continue
                
            elif user_input.lower() == '':
                continue
            
            # Générer la réponse
            print("🤖 Assistant: ", end="", flush=True)
            
            import time
            start_time = time.time()
            response = generate_response(model, tokenizer, user_input)
            generation_time = time.time() - start_time
            
            print(response)
            print(f"⏱️ Temps: {generation_time:.2f}s")
            
            # Mettre à jour les statistiques
            question_count += 1
            total_time += generation_time
            
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir! Merci d'avoir testé le modèle FSBM!")
            if question_count > 0:
                avg_time = total_time / question_count
                print(f"📊 Statistiques: {question_count} questions, temps moyen: {avg_time:.2f}s")
            break
            
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("💡 Essayez une autre question ou tapez 'help' pour voir les exemples")

def main():
    """Fonction principale"""
    try:
        interactive_chat()
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        print("💡 Vérifiez que le modèle est correctement installé")

if __name__ == "__main__":
    main()
