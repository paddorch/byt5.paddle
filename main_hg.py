import argparse

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_dataset

from utils import print_dict


DATASET_CONFIG = {
    'gem-xsum': ['gem', 'xsum']
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='google/byt5-small')
    parser.add_argument('--dataset', type=str, default='gem-xsum')
    args = parser.parse_args()

    print_dict(args.__dict__, name="Command-Line Arguments")

    device = torch.device(args.device)

    print('loading dataset')
    dataset = load_dataset(*DATASET_CONFIG[args.dataset])
    print(dataset)

    print('loading model')
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = model.to(device)

    model_inputs = tokenizer(["Translate English to Frensh: Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt").to(device)
    labels = tokenizer(["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt").input_ids.to(device)

    loss = model(**model_inputs, labels=labels).loss
    print(loss)
