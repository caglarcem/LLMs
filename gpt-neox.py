import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import required libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Step 1: Load the GPT-NeoX model and tokenizer
model_name = "EleutherAI/gpt-neox-20b"  # GPT-NeoX model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Step 2: Define the ICAM analysis input
incident_data = {
    "witness_statements": "I was operating the forklift and didn't notice Sara walking through the aisle. The forklift collided with her when I turned.",
    "follow_up_questions": "The warehouse was busy that day, and visibility was limited due to the placement of pallets.",
    "event_description": "A forklift collision occurred at 9:45 AM in the warehouse when the operator didn't see the pedestrian crossing. The pedestrian sustained injuries."
}

# Combine all the data into a single string for analysis
combined_incident_data = f"""
Incident Description: {incident_data['event_description']}

Witness Statements: {incident_data['witness_statements']}

Follow-up Interview: {incident_data['follow_up_questions']}

Please provide a detailed ICAM safety analysis, including:
1. The contributing factors
2. Root causes
3. Recommendations to prevent this type of incident from reoccurring.
"""

# Step 3: Tokenize the input text
inputs = tokenizer(combined_incident_data, return_tensors="pt", truncation=True, max_length=1024).to('cuda')

# Step 4: Generate the analysis using GPT-NeoX with optimal tuning
output = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],  # Ensures the model focuses on the input
    max_length=600,  # Extend the length for detailed reasoning
    num_beams=5,  # Beam search for better exploration of possible outputs
    no_repeat_ngram_size=3,  # Avoid repetition of phrases
    temperature=0.7,  # Lower temperature for more focused output   
    top_k=50,  # Consider only the top 50 tokens at each step
    top_p=0.9,  # Use nucleus sampling for diversity in generation
    early_stopping=True  # Stop when the model produces a coherent output
)

# Step 5: Decode and print the generated text
analysis = tokenizer.decode(output[0], skip_special_tokens=True)
print("ICAM Analysis:")
print(analysis)