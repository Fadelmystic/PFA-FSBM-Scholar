# Script de conversion du fichier TXT vers dataset Hugging Face
# =============================================================

import json
import re

def convert_txt_to_json_dataset(input_file, output_file):
    """
    Convert a text file with Q&A pairs to a JSON dataset format.
    The text file should have questions starting with "Question : " and 
    answers starting with "Réponse : ".
    """
    
    dataset = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by "Question : " to get individual Q&A pairs
    # Skip the first empty element if the file starts with "Question : "
    qa_pairs = content.split("Question : ")[1:] if content.startswith("Question : ") else content.split("Question : ")
    
    for pair in qa_pairs:
        if not pair.strip():
            continue
            
        # Split by "Réponse : " to separate question and answer
        parts = pair.split("Réponse : ", 1)
        
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            
            # Clean up any extra whitespace and newlines
            question = re.sub(r'\n+', ' ', question).strip()
            answer = re.sub(r'\n+', ' ', answer).strip()
            
            # Add to dataset
            dataset.append({
                "prompt": question,
                "response": answer
            })
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion completed!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total Q&A pairs extracted: {len(dataset)}")
    
    # Show a few examples
    if dataset:
        print("\nFirst 3 examples:")
        for i, item in enumerate(dataset[:3]):
            print(f"\nExample {i+1}:")
            print(f"Prompt: {item['prompt'][:100]}...")
            print(f"Response: {item['response'][:100]}...")

if __name__ == "__main__":
    input_file = "data-chat.txt"
    output_file = "fsbm_qa_dataset.json"
    
    try:
        convert_txt_to_json_dataset(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found. Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
