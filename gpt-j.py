import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import the necessary libraries
from transformers import GPTJForCausalLM, GPT2Tokenizer

# Step 1: Load the GPT-J model and tokenizer
model_name = 'EleutherAI/gpt-j-6B'  # GPT-J model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)














# Set the pad_token to the eos_token
tokenizer.pad_token = tokenizer.eos_token

# Step 2: Define the incident sample data (ICAM analysis input)
incident_data = {
    "witness_statements": "I was operating the forklift and didn't notice Sara walking through the aisle. The forklift collided with her when I turned.",
    "follow_up_questions": "The warehouse was busy that day, and visibility was limited due to the placement of pallets.",
    "event_description": "A forklift collision occurred at 9:45 AM in the warehouse when the operator didn't see the pedestrian crossing. The pedestrian sustained injuries."
}

# Modify the prompt to explicitly ask for an analysis
combined_incident_data = f"""
Incident Description: {incident_data['event_description']}

Witness Statements: {incident_data['witness_statements']}

Follow-up Interview: {incident_data['follow_up_questions']}

Please provide a detailed ICAM safety analysis, including:
1. A description of the incident.
2. The contributing factors and root causes.
3. Recommendations to prevent this type of incident from reoccurring.
"""

# Step 3: Tokenize the input data with an attention mask, using the eos_token as the padding token
inputs = tokenizer(combined_incident_data, return_tensors="pt", max_length=512, truncation=True, padding=True)
attention_mask = inputs['attention_mask']  # Extract the attention mask

# Step 4: Generate the analysis using GPT-J, adjusting generation parameters
summary_ids = model.generate(
    inputs['input_ids'],
    attention_mask=attention_mask,  # Pass the attention mask
    max_length=500,  # Allow more room for the model to generate detailed text
    num_beams=2,  # Less restrictive, allowing for more creative responses
    temperature=0.9,  # Add a little randomness to encourage diverse outputs
    no_repeat_ngram_size=2,  # Avoid repeating phrases
    early_stopping=True
)

# Step 5: Decode and print the output
analysis = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("ICAM Analysis:")
print(analysis)
