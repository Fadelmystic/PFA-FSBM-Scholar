# LLaMA Working - Solution Définitive pour Kaggle avec PEFT
# ========================================================

import os
import torch
import time
import json
import subprocess
import sys

# Configuration
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Vérification et installation de bitsandbytes pour Kaggle
print("🔧 Vérification de bitsandbytes...")
try:
    import bitsandbytes
    print("✅ bitsandbytes déjà installé!")
except ImportError:
    print("📦 Installation de bitsandbytes...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "bitsandbytes"])
        print("✅ bitsandbytes installé avec succès!")
        print("🔄 Redémarrage du runtime nécessaire...")
        print("⚠️ Veuillez redémarrer le runtime Kaggle et relancer ce script")
        raise RuntimeError("Redémarrage du runtime Kaggle nécessaire après installation de bitsandbytes")
    except Exception as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        raise

# Installation de PEFT si nécessaire
print("🔧 Vérification de PEFT...")
try:
    import peft
    print("✅ PEFT déjà installé!")
except ImportError:
    print("📦 Installation de PEFT...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "peft"])
    print("✅ PEFT installé!")

# Maintenant on peut importer les modules
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
        DataCollatorForLanguageModeling, BitsAndBytesConfig
    )
    from datasets import Dataset, load_dataset
    from huggingface_hub import login
    from peft import LoraConfig, get_peft_model, TaskType
    print("✅ Tous les modules importés avec succès!")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    raise

# Configuration LLaMA avec optimisations mémoire
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Version chat plus optimisée
OUTPUT_DIR = "fsbm-llama-working"
MAX_LENGTH = 256  # Plus long pour de meilleures réponses
BATCH_SIZE = 1    # Batch minimal
LEARNING_RATE = 1e-4  # Learning rate plus bas pour plus d'époques
NUM_EPOCHS = 8   # Plus d'époques pour un meilleur fine-tuning

# Token Hugging Face
HF_TOKEN = "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"

# Authentification
print("🔐 Authentification Hugging Face...")
login(token=HF_TOKEN)
print("✅ Authentification réussie!")

# GPU check et configuration mémoire
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🧹 Mémoire GPU nettoyée")
    
    # Configuration mémoire avancée - FORCER cuda:0
    torch.cuda.set_device(0)
    print("🎯 GPU 0 configuré")
    
    # Forcer CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("🔒 CUDA_VISIBLE_DEVICES fixé à 0")

# Charger les données depuis Hugging Face
print("📂 Chargement des données depuis Hugging Face...")
dataset_hf = load_dataset("faduul/fsbm-qa-dataset")
data = dataset_hf["train"]
print(f"✅ {len(data)} exemples chargés")

# Format LLaMA optimisé
def format_llama_data(data):
    formatted = []
    for item in data:
        prompt = item["prompt"].strip()
        response = item["response"].strip()
        # Format LLaMA chat optimisé
        text = f"<s>[INST] {prompt} [/INST] {response} </s>"
        formatted.append({"text": text})
    return formatted

# Préparation des données
print("🔄 Formatage des données...")
formatted_data = format_llama_data(data)
dataset = Dataset.from_list(formatted_data)
print(f"✅ Données formatées: {len(formatted_data)} exemples")

# Tokenizer avec token
print(f"🔧 Initialisation tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

print("🔤 Tokenisation...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
print("✅ Tokenisation terminée")

# Split train/validation
dataset_size = len(tokenized_dataset)
train_size = int(0.9 * dataset_size)
train_dataset = tokenized_dataset.select(range(train_size))
val_dataset = tokenized_dataset.select(range(train_size, dataset_size))
print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# Configuration 8-bit pour éviter l'erreur CUBLAS
print("🔧 Configuration 8-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Utiliser 8-bit au lieu de 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Modèle avec optimisations mémoire
print("🚀 Chargement du modèle LLaMA avec optimisations...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map={"": 0},  # FORCER tout sur cuda:0
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Configuration LoRA pour PEFT
print("🔧 Configuration LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank
    lora_alpha=32,  # Alpha parameter
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

# Appliquer LoRA au modèle
print("🔧 Application de LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments ultra-optimisés pour LLaMA avec PEFT
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=20,  # Plus de warmup pour plus d'époques
    logging_steps=10,  # Plus de logging pour suivre le progrès
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # Garder plus de checkpoints
    fp16=True,
    remove_unused_columns=False,
    report_to=[],
    gradient_accumulation_steps=8,  # Moins d'accumulation pour plus de mises à jour
    optim="paged_adamw_8bit",  # Optimiseur 8-bit
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    gradient_checkpointing=False,  # Désactivé pour éviter les conflits de device
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False,
    save_safetensors=False,
    torch_compile=False,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Entraînement
print("🎯 Démarrage entraînement LLaMA avec PEFT...")
start_time = time.time()
trainer.train()
training_time = time.time() - start_time

# Sauvegarder
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n📊 RÉSULTATS:")
print(f"⏱️ Temps: {training_time:.1f}s")
print(f"✅ Modèle sauvegardé: {OUTPUT_DIR}")

# Test simple
print(f"\n🧪 Test du modèle LLaMA...")
test_model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    quantization_config=bnb_config,
    device_map={"": 0},  # FORCER tout sur cuda:0
    trust_remote_code=True
)
test_model.eval()

def test_generation(prompt):
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    
    # FORCER sur cuda:0
    if torch.cuda.is_available():
        inputs = inputs.to("cuda:0")
    
    with torch.no_grad():
        outputs = test_model.generate(
            inputs,
            max_new_tokens=100,  # Plus de tokens pour des réponses plus longues
            do_sample=True,      # Activation du sampling pour plus de variété
            temperature=0.7,     # Température pour la créativité
            top_p=0.9,           # Top-p sampling
            repetition_penalty=1.1,  # Éviter la répétition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()
    return response

# Tests
test_questions = [
    "Qu'est-ce que le Big Data?",
    "Expliquez Python",
    "Qu'est-ce que l'IA?"
]

print("\n🧪 TESTS LLaMA:")
for i, question in enumerate(test_questions, 1):
    print(f"\n--- Test {i} ---")
    print(f"📝 Question: {question}")
    response = test_generation(question)
    print(f"🤖 Réponse: {response}")

print(f"\n🎉 LLaMA WORKING TERMINÉ!")
print(f"💾 Optimisations appliquées:")
print(f"- 8-bit quantization (évite erreur CUBLAS)")
print(f"- PEFT LoRA adapters")
print(f"- Gradient accumulation 8x")
print(f"- Paged AdamW 8-bit")
print(f"- Gradient checkpointing désactivé")
print(f"- Low CPU memory usage")
print(f"- 3 époques de fine-tuning")
print(f"- Séquences de 256 tokens")
print(f"- Génération avec sampling créatif")
print(f"✅ LLaMA fonctionne maintenant!")
