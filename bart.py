# Test mBART large
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Import necessary libraries
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Set environment variable to avoid OpenMP errors (Optional)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Step 1: Load the mBART model and tokenizer
model_name = 'facebook/mbart-large-50-many-to-many-mmt'  # mBART model for multilingual tasks
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

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

Provide a detailed ICAM safety analysis of this event, including contributing factors, root causes, and recommendations.
"""

# Step 3: Tokenize the input data
inputs = tokenizer(combined_incident_data, return_tensors="pt", max_length=1024, truncation=True)

# Step 4: Generate the analysis in English using the forced_bos_token_id for English (en_XX)
summary_ids = model.generate(
    inputs['input_ids'],
    max_length=300,
    num_beams=4,
    early_stopping=True,
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]  # Ensure output is in English
)

# Step 5: Decode and print the output (in English)
analysis_en = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("ICAM Analysis in English:")
print(analysis_en)

# Optional Step 6: Translate the output to another language (e.g., Spanish)
translated_inputs = tokenizer(analysis_en, return_tensors="pt")
translated_ids = model.generate(
    translated_inputs['input_ids'],
    forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]  # Ensure translation is to Spanish
)

# Decode and print the output in Spanish
analysis_es = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
print("\nICAM Analysis in Spanish:")
print(analysis_es)