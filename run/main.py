from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
import data.retrive_data as retrive_data
import utils.params as params
import nns.generator as generator
import nns.encoder as encoder
import run.learn as learn 
import numpy as np
import random
import os

if __name__ == '__main__':
    args = params.parse_args()
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    torch.cuda.manual_seed_all(args.rand_seed)

    train_data, dev_data, test_data = retrive_data.get_dataloaders(args)
    #get trained models

    if args.train:
        gen, enc = generator.Generator(args), encoder.Encoder(args)
        gen, enc = learn.train(train_data, dev_data, gen, enc, args)
        
    else:
        gen = torch.load(os.path.join(args.save_dir, "_gen_t"))
        enc = torch.load(os.path.join(args.save_dir, "_enc_t"))
    test_stats = learn.test(test_data, gen, enc, args)
