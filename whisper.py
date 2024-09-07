import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import whisper

# Load the Whisper model (you can use 'base', 'small', 'medium', 'large' based on your preference)
model = whisper.load_model("large")

# Transcribe the audio file
result = model.transcribe("C:/repo/Whisper/audio/09042024_RD04.wav")

# Print the transcribed text
print(result['text'])
