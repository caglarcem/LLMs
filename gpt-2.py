import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import the necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2-xl"  # You can also use 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' for larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Define the incident sample data (ICAM analysis input)
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

Provide a detailed ICAM safety analysis, including contributing factors, root causes, and recommendations.
"""

# Step 3: Tokenize the input text for GPT-2
inputs = tokenizer.encode(combined_incident_data, return_tensors='pt')

# Step 4: Generate the analysis using GPT-2
outputs = model.generate(
    inputs,
    max_length=300,  # Set the max length of the generated text
    num_beams=5,     # Use beam search to generate diverse outputs
    no_repeat_ngram_size=2,  # Prevent repeated phrases
    temperature=0.7,  # Adjust the randomness of the generation
    early_stopping=True
)

# Step 5: Decode and print the output
analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("ICAM Analysis:")
print(analysis)
