import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_dataset

from utils import print_dict


DATASET_CONFIG = {
    'gem-xsum': ['gem', 'xsum']
}


DATASET_FIELDS = {
    'gem-xsum': {
        'input': 'document',
        'target': 'target'
    }
}


def train(model: torch.nn.Module, device, optimizer: torch.optim.Optimizer, 
          train_loader: DataLoader, val_loader: DataLoader):
    model.train()
    tot_loss = 0.0
    tot_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()

        tot_loss += loss.item()
        tot_samples += batch['input_ids'].size()[0]

    return tot_loss / tot_samples


def main(args: argparse.Namespace):
    print_dict(args.__dict__, name="Command-Line Arguments")
    device = torch.device(args.device)

    print('loading dataset')
    dataset = load_dataset(*DATASET_CONFIG[args.dataset])
    print(dataset)

    print('loading model')
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("===== Model Config =====")
    print(model.config)
    print("========================")

    model = model.to(device)

    def tokenize(samples):
        text_field = DATASET_FIELDS[args.dataset]['input']
        target_field = DATASET_FIELDS[args.dataset]['target']
        model_inputs = tokenizer(samples[text_field], max_length=args.max_source_length, padding="max_length", truncation=True)
        labels = tokenizer(samples[target_field], max_length=args.max_source_length, padding="max_length", truncation=True).input_ids
        return {
            **model_inputs,
            'labels': labels
        }

    print('preprocessing dataset')
    preprocessed_dataset = dataset.map(tokenize, batched=True, num_proc=8)
    VALID_COLUMNS = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    columns_to_remove = [c for c in preprocessed_dataset['train'].column_names if c not in VALID_COLUMNS]
    print(f'columns to remove: {columns_to_remove}')
    preprocessed_dataset = preprocessed_dataset.remove_columns(columns_to_remove)
    preprocessed_dataset.set_format('torch')
    print(f'current dataset columns: {preprocessed_dataset.column_names}')
    train_dataset, test_dataset, val_dataset = [preprocessed_dataset[x] for x in ['train', 'test', 'validation']]
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=data_collator)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_inputs = tokenizer(["Translate English to Frensh: Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt").to(device)
    labels = tokenizer(["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt").input_ids.to(device)

    loss = model(**model_inputs, labels=labels).loss
    print(loss)

    for epoch in range(args.num_epochs):
        loss = train(model, device, optimizer, train_loader, val_loader)

        print(f'epoch {epoch:05d}, loss {loss:.8f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default='google/byt5-small')
    parser.add_argument('--dataset', type=str, default='gem-xsum')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--max_source_length', type=int, default=1024)
    args = parser.parse_args()

    main(args)