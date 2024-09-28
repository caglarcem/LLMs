import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Vicuna model and tokenizer (assumed to be hosted on Hugging Face)
model_name = "lmsys/vicuna-13b-v1.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

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

# Tokenize the input text
inputs = tokenizer(combined_incident_data, return_tensors="pt")

# Generate text from the model
output = model.generate(inputs['input_ids'], max_length=300, num_beams=4, temperature=0.7, early_stopping=True)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Vicuna Analysis:")
print(generated_text)
