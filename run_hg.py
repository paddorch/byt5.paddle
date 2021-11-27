import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

device = torch.device('cuda:0')
model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')


def run_model(prompt: str) -> str:
    model_inputs = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    print(model_inputs)


model_inputs = tokenizer(["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt")
labels = tokenizer(["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt").input_ids

loss = model(**model_inputs, labels=labels).loss
