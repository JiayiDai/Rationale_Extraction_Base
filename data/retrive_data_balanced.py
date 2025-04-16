import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import numpy as np

def get_dataloaders(args, train_split=0.8):
    if args.dataset == 'imdb':
        args.num_class = 2
        dataset = load_dataset('imdb', split=['train', 'test'])
    else:
        raise Exception("Unknown dataset!")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load and verify original dataset balance
    original_train = dataset[0]
    original_labels = np.array(original_train['label'])
    print(f"Original class distribution: {np.unique(original_labels, return_counts=True)}")

    # Perform stratified split with balance verification
    stratified_split = original_train.train_test_split(
        test_size=1 - train_split,
        stratify_by_column='label',
        seed=args.rand_seed
    )

    # Verify split balance
    def check_balance(dataset, name):
        labels = np.array(dataset['label'])
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{name} class distribution: {dict(zip(unique, counts))}")
        assert np.abs(counts[0] - counts[1]).max() <= 1, "Class imbalance detected!"

    check_balance(stratified_split['train'], "Training")
    check_balance(stratified_split['test'], "Validation")

    # Tokenization after splitting
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )

    train_data = stratified_split['train'].map(
        tokenize_function,
        batched=True,
        num_proc=8
    )
    val_data = stratified_split['test'].map(
        tokenize_function,
        batched=True,
        num_proc=8
    )
    test_data = dataset[1].map(
        tokenize_function,
        batched=True,
        num_proc=8
    )

    # Collate function remains the same
    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        tokenized_texts = [tokenizer.convert_ids_to_tokens(item['input_ids']) for item in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels,
            'text': tokenized_texts
        }

    # Create dataloaders with balance verification
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, test_dataloader