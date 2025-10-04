from transformers import pipeline
generator = pipeline("text-generation", model="distilgpt2")
print(generator("Hello IRCTC", max_new_tokens=50))
