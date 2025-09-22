#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Rapide CPU - Modèle FSBM LLaMA Fine-tuné
==============================================

Test rapide du modèle adapté pour CPU (sans GPU CUDA).
"""

import os
import torch
import time

# Configuration
MODEL_PATH = "Model"
HF_TOKEN = "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"

def quick_test_cpu():
    """Test rapide du modèle sur CPU"""
    print("🚀 TEST RAPIDE CPU - MODÈLE FSBM LLaMA")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import login
        
        # Configuration
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print("🔐 Authentification...")
        login(token=HF_TOKEN)
        
        print("📝 Chargement tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("🧠 Chargement modèle (CPU)...")
        # Charger le modèle sans quantization pour CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cpu",  # Forcer CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32  # Utiliser float32 pour CPU
        )
        
        print("✅ Modèle chargé sur CPU! Test en cours...\n")
        
        # Questions de test rapide
        test_questions = [
            "Qu'est-ce que le Big Data?",
            "Expliquez Python",
            "Qu'est-ce que l'IA?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"--- Test {i} ---")
            print(f"Q: {question}")
            
            # Générer réponse
            formatted_prompt = f"<s>[INST] {question} [/INST]"
            inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
            
            # Forcer sur CPU
            inputs = inputs.to("cpu")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "[/INST]" in response:
                response = response.split("[/INST]")[1].strip()
            
            generation_time = time.time() - start_time
            
            print(f"R: {response}")
            print(f"⏱️ {generation_time:.2f}s\n")
        
        print("🎉 TEST RÉUSSI! Le modèle fonctionne sur CPU!")
        print("💡 Utilisez 'python interactive_chat_cpu.py' pour un chat complet")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Vérifiez que le modèle est correctement installé")

if __name__ == "__main__":
    quick_test_cpu()
