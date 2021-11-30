import os
import os.path as osp
import argparse
from tqdm import tqdm
from happy_config import ConfigLoader
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.optimization import Adafactor
from datasets import load_dataset, load_metric

from utils import print_dict
from hc_config import ExpConfig


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
    optimizer.zero_grad()
    tot_loss = 0.0
    tot_samples = 0
    acc_tokens = 0

    with tqdm(total=len(train_loader), desc=f'[TRAIN] epoch {epoch:05d}') as pbar:
        for batch in train_loader:

            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss.mean()
            loss.backward()

            num_batch_tokens = batch['attention_mask'].sum().item()
            acc_tokens += num_batch_tokens
            if acc_tokens >= config.opt.num_tokens_per_batch:
                acc_tokens = 0
                optimizer.step()
                optimizer.zero_grad()

            tot_loss += loss.item()
            tot_samples += batch['input_ids'].size()[0]
            pbar.update(1)
            pbar.set_postfix({'loss': f'{tot_loss / tot_samples:.4f}', 'tk': acc_tokens})

    return tot_loss / tot_samples


def evaluate(model: torch.nn.Module, tokenizer: AutoTokenizer, 
             eval_loader: DataLoader, metric, epoch: int = 0):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    model.eval()
    device = list(model.parameters())[0].data.device

    with tqdm(total=len(eval_loader), desc=f'[EVAL] epoch {epoch:05d}') as pbar:
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.generate(batch['input_ids'], max_length=config.seq.max_target_length)
            outputs = outputs.cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            preds, refs = postprocess_text(preds, refs)

            metric.add_batch(predictions=preds, references=refs)
            pbar.update(1)

    return metric.compute()


def main(config: ExpConfig):
    print_dict(asdict(config), name="Command-Line Arguments")
    device = torch.device(config.device)

    print('loading dataset')
    dataset = load_dataset(*DATASET_CONFIG[config.dataset])
    print(dataset)

    print('loading model')
    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    print("===== Model Config =====")
    print(model.config)
    print("========================")

    model = model.to(device)

    def tokenize(samples):
        text_field = DATASET_FIELDS[config.dataset]['input']
        target_field = DATASET_FIELDS[config.dataset]['target']
        model_inputs = tokenizer(samples[text_field], max_length=config.seq.max_source_length, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(samples[target_field], max_length=config.seq.max_target_length, padding="max_length", truncation=True).input_ids
        return {
            **model_inputs,
            'labels': labels
        }

    print('preprocessing dataset')
    preprocessed_dataset = dataset.map(tokenize, batched=True, num_proc=8)
    KEEP_COLUMNS = ['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    columns_to_remove = [c for c in preprocessed_dataset['train'].column_names if c not in KEEP_COLUMNS]
    print(f'columns to remove: {columns_to_remove}')
    preprocessed_dataset = preprocessed_dataset.remove_columns(columns_to_remove)
    preprocessed_dataset.set_format('torch')
    print(f'current dataset columns: {preprocessed_dataset.column_names}')
    train_dataset, test_dataset, val_dataset = [preprocessed_dataset[x] for x in ['train', 'test', 'validation']]

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.opt.step_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.opt.step_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.opt.step_size)

    optimizer = Adafactor(model.parameters(), lr=config.opt.lr, beta1=config.opt.beta1, relative_step=False)

    metric = load_metric('sacrebleu')

    unwrapped_model = model
    model = nn.DataParallel(model)

    best_val_score = 0.0
    best_val_epoch = 1
    for epoch in range(1, config.opt.num_epochs + 1):
        loss = train(model, optimizer, train_loader, epoch)

        val_result = evaluate(unwrapped_model, tokenizer, val_loader, metric)
        val_score = val_result['score']

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_epoch = epoch

        print(f'epoch {epoch:05d}, loss {loss:.8f}, val {val_score:.4f}, best val {best_val_score:.4f}')
        torch.save(unwrapped_model.state_dict(), osp.join(config.chkpt_dir, f'model_{epoch}.pt'))


if __name__ == '__main__':
    loader = ConfigLoader(ExpConfig, config="params/default.yml")
    config = loader()

    os.makedirs(config.chkpt_dir, exist_ok=True)

    main(config)
