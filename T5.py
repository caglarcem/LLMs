# Test mBART large
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import necessary libraries
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Set environment variable to avoid OpenMP errors (Optional)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1: Load the mT5 model and tokenizer
model_name = 'google/mt5-small'  # You can also use mt5-base, mt5-large, etc.
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Step 2: Define the incident sample data (ICAM analysis input)
incident_data = {
    "witness_statements": "I was operating the forklift and didn't notice Sara walking through the aisle. The forklift collided with her when I turned.",
    "follow_up_questions": "The warehouse was busy that day, and visibility was limited due to the placement of pallets.",
    "event_description": "A forklift collision occurred at 9:45 AM in the warehouse when the operator didn't see the pedestrian crossing. The pedestrian sustained injuries."
}

# Combine all the data into a single string for analysis and add a task definition
combined_incident_data = f"""
summarize: Incident Description: {incident_data['event_description']}

Witness Statements: {incident_data['witness_statements']}

Follow-up Interview: {incident_data['follow_up_questions']}

Provide a detailed ICAM safety analysis of this event, including contributing factors, root causes, and recommendations.
"""

# Step 3: Tokenize the input data with proper task instruction
inputs = tokenizer(combined_incident_data, return_tensors="pt", max_length=512, truncation=True)

# Step 4: Generate the analysis
summary_ids = model.generate(
    inputs['input_ids'], 
    max_length=300, 
    num_beams=4, 
    early_stopping=True
)

# Step 5: Decode and print the output (in English)
analysis = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("ICAM Analysis:")
print(analysis)

# Optional: Translate the ICAM analysis to another language (e.g., Spanish)
translated_input = tokenizer(analysis, return_tensors="pt")
translated_ids = model.generate(translated_input['input_ids'], max_length=300)

# Decode the translated text
translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
print("\nTranslated ICAM Analysis:")
print(translated_text)