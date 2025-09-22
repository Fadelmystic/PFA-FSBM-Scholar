#!/usr/bin/env python3
"""
Script pour uploader le nouveau dataset structuré sur Hugging Face
Remplace l'ancien dataset faduul/chatbot-data
"""

import json
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi

def upload_dataset_to_huggingface(json_file, dataset_name, token):
    """
    Upload a JSON dataset to Hugging Face Hub
    """
    
    print(f"📂 Loading dataset from: {json_file}")
    
    # Load the JSON dataset
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Dataset loaded with {len(data)} examples")
    
    # Create Hugging Face dataset
    dataset = Dataset.from_list(data)
    dataset_dict = DatasetDict({"train": dataset})
    
    print(f"✅ Dataset created successfully")
    
    # Login to Hugging Face with token
    print(f"🔐 Logging in to Hugging Face...")
    login(token=token)
    
    # Upload to Hugging Face Hub
    print(f"🚀 Uploading dataset to: {dataset_name}")
    
    try:
        dataset_dict.push_to_hub(
            dataset_name,
            private=False,
            token=token
        )
        print(f"🎉 Dataset uploaded successfully!")
        print(f"📎 Dataset URL: https://huggingface.co/datasets/{dataset_name}")
        return True
    except Exception as e:
        print(f"❌ Error uploading dataset: {str(e)}")
        return False

def main():
    # Configuration
    json_file = "fsbm_qa_dataset.json"
    dataset_name = "faduul/fsbm-qa-dataset"  # Using your username
    token = "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"
    
    print("🔄 UPLOADING DATASET TO HUGGING FACE")
    print("=" * 50)
    
    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"❌ File {json_file} not found!")
        return
    
    # Upload dataset
    success = upload_dataset_to_huggingface(json_file, dataset_name, token)
    
    if success:
        print(f"\n✅ SUCCESS!")
        print(f"Dataset is now available at: https://huggingface.co/datasets/{dataset_name}")
        print(f"You can use it in your code with: datasets.load_dataset('{dataset_name}')")
    else:
        print(f"\n❌ Upload failed. Please check the error message above.")

if __name__ == "__main__":
    main()
