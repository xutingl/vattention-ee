from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_extra import *
# Load the model with balcony exits
model_path = "workspace/downloaded_models/models--parsakaveh--Balcony-LLaMA2-7B"
exit_layer = 15 # The layer index of desired exit balcony
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, output_exit_layers=exit_layer)

# Generate text with early exits
input_text = "Translate the following English text to French: 'Hello, how are you?'"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))