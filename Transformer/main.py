# https://huggingface.co/learn/llm-course/chapter2/2?fw=pt

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

input=["I am felling happy today","I will break your teeth into peices"]
inputs=tokenizer(input,return_tensors="pt",padding=True,truncation=True)
print(inputs)



from transformers import AutoModelForSequenceClassification
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

outputs=model(**inputs)
print(outputs.logits.shape)

import torch
prediction=torch.nn.functional.softmax(outputs.logits,dim=-1)
print(prediction)

print(model.config.id2label)