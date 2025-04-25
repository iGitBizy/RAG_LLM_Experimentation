
# updates


import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.ai.models.mistral.utils import find_directory, CURRENT_MODEL, supported_models
import evaluate
from tqdm import tqdm
import pandas as pd

######################### SET UP PHASE #######################################
# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load model and tokenizer
try:
    # Get the Hugging Face model ID from the supported_models dictionary
    model_id = supported_models.get(CURRENT_MODEL, CURRENT_MODEL)
    
    # First try finding the model in the local directory structure
    cache_dir = find_directory('mistral')
    model_dir = os.path.join(cache_dir, CURRENT_MODEL) if cache_dir else None
    
    # If found locally, use that path; otherwise use the model ID from Hugging Face
    model_path = model_dir if model_dir and os.path.exists(model_dir) else model_id
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    logging.info(f"Loaded {CURRENT_MODEL} model successfully")
    
except Exception as e:
    logging.error(f"Failed to load Mistral model: {e}")
    tokenizer = None
    model = None


def run_mistral_query(query_text: str) -> str:
    """
    Process a query through the Mistral model and return the generated response.
    
    Args:
        query_text (str): The input query to process
        
    Returns:
        str: The model's response to the query
    """
    try:
        if model is None or tokenizer is None:
            raise ValueError("Model or tokenizer not loaded correctly")
            
        messages = [
            {"role": "user", "content": query_text}
        ]
        
        # Set EOS token as pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        # Format input in the Mistral chat format with attention mask
        input_ids = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt",
            padding=True, #Enable padding
            max_length = 32768, #set max lenght for truncation
            truncation=True, # Handle longer inputs
            return_attention_mask=True  # Explicitly request attention mask
        ).to(model.device)
        
        # Generate response with explicit pad token setting
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id # Explicity set pad token
            )
            
        # Decode and return response, skipping the input
        response = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        return response.strip()
        
    except Exception as exc:
        logging.error(f"Error processing query with Mistral: {exc}")
        logging.exception(exc)
        raise exc


######################### Print model configuration #########################
print("\nModel Configuration:")
print("-" * 50)
print(model.config)

# Access tokenizer configuration
print("\nTokenizer Configuration:")
print("-" * 50)
# print(tokenizer.model_config)

# Print special tokens
print("\nSpecial Tokens:")
print("-" * 50)
print(f"bos_token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
print(f"eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"unk_token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
print(f"mask_token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

# Print tokenizer parameters
print("\nTokenizer Parameters:")
print("-" * 50)
print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Padding side: {tokenizer.padding_side}")
print(f"Truncation side: {tokenizer.truncation_side}")
print("\nModel is on:", model.device)


##################### Evaluation Phase ######################################

# define evaluation function for collecting predictions
def evaluate_model(eval_dataset):
    predictions, references, queries = [], [], []
    
    for item in tqdm(eval_dataset, desc="Evaluating Model Responses"):
        query = item["query"]
        reference = item["reference"]
        response = run_mistral_query(query)
        
        
        queries.append(query)
        predictions.append(response)
        references.append(reference)


    return predictions, references, queries 
    
    
# Prepare dataset 
eval_dataset = [
    {"query": "What is the capital of France?", "reference": "The capital of France is Paris."},
    {"query": "Explain Newton's second law.", "reference": "Newton's second law states that force equals mass times acceleration."}
    ]

# Run model and collect predictions
predictions, references, queries = evaluate_model(eval_dataset)


# Evaluate using metrics 

# ROUGE SCORE   
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)

# BERTScore 
bertscore = evaluate.load("bertscore")
bertscore_result = bertscore.compute(
    predictions=predictions, references=references, lang="en")

print("ROUGE Results:", rouge_result)
print("BERTScore Results:", {
    "precision": sum(bertscore_result["precision"]) / len(bertscore_result["precision"]),
    "recall": sum(bertscore_result["recall"]) / len(bertscore_result["recall"]),
    "f1": sum(bertscore_result["f1"]) / len(bertscore_result["f1"])
})


######## OUTPUT RESULTS TO CSV FILE #####################################
df_results = pd.DataFrame({
    "query": [item["query"] for item in eval_dataset],
    "reference": references,
    "predictions": predictions
})

df_results.to_csv("mistral_evaluation_results_v2.csv", index=False)






