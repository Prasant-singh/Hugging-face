# https://huggingface.co/learn/llm-course/chapter2/2?fw=pt

from transformers import AutoTokenizer

# Getting the exact same tokenizer that model was trained with
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Converting input to tokens to pass to the model
input=["I am felling happy today","I will break your teeth into peices"]
inputs=tokenizer(input,return_tensors="pt",padding=True,truncation=True)
print("Inputs:", inputs)


# Importing the model And using Auto model class so that it can automatically finds the model that best fit with situation
from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Passing the inputs to the model FORWARD PASS
outputs=model(**inputs)
print("Size of the image pass to the HEAD layer:", outputs.last_hidden_state.shape)
# Cheching shape of logits. In short cheching the shape of logits that HEAD passes to the output layer
print(outputs.logits.shape)
import torch
prediction=torch.nn.functional.softmax(outputs.logits,dim=-1)
print("Prediction:", prediction)

print("Label Mapping:", model.config.id2label)
