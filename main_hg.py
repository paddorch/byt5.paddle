import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_dataset, load_metric

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


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
          train_loader: DataLoader, epoch: int):
    model.train()
    tot_loss = 0.0
    tot_samples = 0

    with tqdm(total=len(train_loader), desc=f'[TRAIN] epoch {epoch:05d}') as pbar:
        for batch in train_loader:
            optimizer.zero_grad()

            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()

            optimizer.step()

            tot_loss += loss.item()
            tot_samples += batch['input_ids'].size()[0]
            pbar.update(1)
            pbar.set_postfix({'loss': f'{tot_loss / tot_samples:.4f}'})

    return tot_loss / tot_samples


def evaluate(model: torch.nn.Module, tokenizer: AutoTokenizer, eval_loader: DataLoader, metric):
    model.eval()
    device = list(model.parameters())[0].data.device
    for batch in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.generate(batch['input_ids'], max_length=1024)
        predicted = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
        references = [tokenizer.decode(x, skip_special_tokens=True) for x in batch['labels'].cpu().tolist()]


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
        with tokenizer.as_target_tokenizer():
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

    metric = load_metric('bleu')

    unwrapped_model = model
    model = nn.DataParallel(model)

    evaluate(unwrapped_model, tokenizer, val_loader, metric)

    # model_inputs = tokenizer(["Translate English to Frensh: Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt").to(device)
    # labels = tokenizer(["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt").input_ids.to(device)

    # loss = model(**model_inputs, labels=labels).loss
    # print(loss)

    for epoch in range(args.num_epochs):
        loss = train(model, optimizer, train_loader, epoch)

        print(f'epoch {epoch:05d}, loss {loss:.8f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='google/byt5-small')
    parser.add_argument('--dataset', type=str, default='gem-xsum')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--max_source_length', type=int, default=1024)
    args = parser.parse_args()

    main(args)
