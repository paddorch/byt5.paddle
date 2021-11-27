import argparse
from re import A
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer


def print_dict(d: dict, name: str = None):
    if name is not None:
        print(f'===== {name} =====')
    for k, v in d.items():
        print(f'{k} = {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='google/byt5-small')
    args = parser.parse_args()

    print_dict(args.__dict__, name="Command-Line Arguments")

    device = torch.device(args.device)

    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = model.to(device)

    model_inputs = tokenizer(["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt").to(device)
    labels = tokenizer(["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt").input_ids.to(device)

    loss = model(**model_inputs, labels=labels).loss
