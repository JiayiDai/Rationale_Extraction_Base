import torch
import torch.nn as nn
from transformers import get_scheduler

def get_optimizer(models, args):
    params = []
    for model in models:
        params.extend([param for param in model.parameters() if param.requires_grad])
    return torch.optim.AdamW(params, lr=args.lr)

def get_loss(logit,y):
    loss = nn.CrossEntropyLoss()
    return loss(logit, y)

def lr_scheduler(optimizer, batches, epochs):
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=batches, num_training_steps=epochs*batches)
    return(lr_scheduler)

def get_rationales(mask, text):
    if mask is None:
        return text
    masked_text = []
    for i, t in enumerate(text):
        sample_mask = list(mask.data[i])
        original_words = t#.split()
        words = [ w if m  > .5 else "_" for w,m in zip(original_words, sample_mask) ]
        masked_sample = " ".join(words)
        masked_text.append(masked_sample)
    return masked_text
