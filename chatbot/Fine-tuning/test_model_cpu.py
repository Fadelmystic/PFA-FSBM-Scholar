#!/usr/bin/env python3
"""
Test script for the fine-tuned FSBM Scholar Assistant model - CPU Version
"""

import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
MODEL_PATH = "Model"  # Path to your fine-tuned model
BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = "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"

# Disable warnings
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model():
    """Load the fine-tuned model for CPU"""
    print("🔧 Loading model for CPU...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model without quantization for CPU
    print("🚀 Loading base model (CPU mode)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Load fine-tuned adapter
    print("🔧 Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    
    print("✅ Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=150):
    """Generate a response from the model"""
    # Format prompt for LLaMA chat
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize input
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()
    
    return response

def interactive_chat():
    """Interactive chat interface"""
    print("🎓 FSBM Scholar Assistant - Fine-tuned Model (CPU)")
    print("=" * 50)
    print("Type 'quit' to exit, 'help' for sample questions")
    print()
    
    # Load model
    model, tokenizer = load_model()
    
    # Sample questions from the dataset
    sample_questions = [
        "Qu'est-ce que le Big Data?",
        "Expliquez les 3 V du Big Data",
        "Quelle est la différence entre l'apprentissage supervisé et non supervisé?",
        "Qu'est-ce qu'Apache Hadoop?",
        "Expliquez le concept de validation croisée",
        "Qu'est-ce qu'une API?",
        "Définissez l'ingénierie des caractéristiques",
        "Qu'est-ce que la régularisation en apprentissage automatique?"
    ]
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤔 Vous: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Au revoir!")
                break
            elif user_input.lower() == 'help':
                print("\n📚 Questions d'exemple:")
                for i, question in enumerate(sample_questions, 1):
                    print(f"{i}. {question}")
                continue
            elif user_input.lower() == '':
                continue
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")

def test_sample_questions():
    """Test the model with sample questions"""
    print("🧪 Testing model with sample questions...")
    
    # Load model
    model, tokenizer = load_model()
    
    # Test questions
    test_questions = [
        "Qu'est-ce que le Big Data?",
        "Expliquez les 3 V du Big Data",
        "Quelle est la différence entre l'apprentissage supervisé et non supervisé?",
        "Qu'est-ce qu'Apache Hadoop?",
        "Expliquez le concept de validation croisée"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i} ---")
        print(f"📝 Question: {question}")
        
        response = generate_response(model, tokenizer, question)
        print(f"🤖 Réponse: {response}")
        print("-" * 50)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_sample_questions()
    else:
        interactive_chat()
