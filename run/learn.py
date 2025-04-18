import utils.learn_settings as learn_settings
import tqdm
import utils.metrics as metrics
import torch
import numpy as np
import os, pickle

def train(data_train, data_dev, gen, enc, args):
    if args.cuda:
        gen = gen.cuda()
        enc = enc.cuda()

    args.lr = args.init_lr
    optimizer = learn_settings.get_optimizer([gen, enc], args)
    lr_scheduler = learn_settings.lr_scheduler(optimizer, len(data_train), args.epochs)
    step = 0
    dev_min_loss = 999
    dev_epoch_stats = []
    dev_best_epcoh = 1
    for epoch in range(1, args.epochs+1):
        print("-------------\nEpoch {}:\n".format(epoch))
        for mode, data_loader in [("Train", data_train), ("Dev", data_dev)]:
            epoch_stat, step, _, _ = run_epoch(data_loader, gen, enc, args, mode=="Train", optimizer, lr_scheduler, step)
            print(mode, epoch_stat)
            if mode == "Dev":
                dev_epoch_stats.append(epoch_stat)
                if epoch_stat['loss'] <= dev_min_loss:
                    dev_min_loss = epoch_stat['loss']
                    dev_best_epcoh = epoch
                    if not os.path.isdir(args.save_dir):
                        os.makedirs(args.save_dir)
                    torch.save(gen, os.path.join(args.save_dir, "gen_"+args.task_name))
                    torch.save(enc, os.path.join(args.save_dir, "enc_"+args.task_name))
                print('---- Best Dev {} is {:.4f} at epoch {}'.format(
                    'loss',
                    dev_epoch_stats[dev_best_epcoh-1]['loss'],
                    dev_best_epcoh))
    if os.path.exists(os.path.join(args.save_dir, "gen_"+args.task_name)):
        gen.cpu()
        enc.cpu()
        gen = torch.load(os.path.join(args.save_dir, "gen_"+args.task_name), weights_only=False)
        enc = torch.load(os.path.join(args.save_dir, "enc_"+args.task_name), weights_only=False)
    return(gen, enc)

def test(data_test, gen, enc, args):
    if args.cuda:
        gen = gen.cuda()
        enc = enc.cuda()
    epoch_stat, _, rationales, texts = run_epoch(data_test, gen, enc, args)
    print("test", epoch_stat)
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)
    pickle.dump(epoch_stat, open(os.path.join(args.results_dir, "test_stats"+args.task_name),'ab'))
    pickle.dump(rationales, open(os.path.join(args.results_dir, "test_rationales"+args.task_name),'ab'))
    pickle.dump(texts, open(os.path.join(args.results_dir, "test_texts"+args.task_name),'ab'))
    return(epoch_stat)


def run_epoch(data_loader, gen, enc, args, is_train=False, optimizer=None, lr_scheduler=None, step=None):
    data_iter = data_loader.__iter__()#len=batch number
    losses = []
    pred_losses = []
    select_losses = []
    contig_losses = []
    preds = []
    golds = []
    texts = []
    rationales = []
    if is_train:
        gen.train()
        enc.train()
    else:
        gen.eval()
        enc.eval()
    for batch in tqdm.tqdm(data_iter):#range(len(data_iter))
        if is_train:
            step += 1
            if  step % 100 == 0:
                args.gumbel_temprature = max( np.exp((step+1) *-1* args.gumbel_decay), .5)
        #batch keys: ['input_ids', 'attention_mask', 'label', 'text']
        x = batch["input_ids"]
        text = batch["text"]
        y = batch["label"]
        att_mask = batch["attention_mask"]
        if args.cuda:
            x, y, att_mask = x.cuda(), y.cuda(), att_mask.cuda()
        if args.get_rationales:
            mask, _, _ = gen(x, att_mask=att_mask)
            select_loss = gen.loss(mask)
            select_losses.append(select_loss.item())
            if not is_train:
                rationales.extend(learn_settings.get_rationales(mask, text))
        else:
            mask = None
        logit = enc(x, att_mask=att_mask, mask=mask)
        pred_loss = learn_settings.get_loss(logit, y)
        if args.get_rationales:
            loss = pred_loss + args.select_lambda*select_loss
        else:
            loss = pred_loss

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
        losses.append(loss.item())
        pred_losses.append(pred_loss.item())
        preds.extend(torch.argmax(logit, dim=-1).cpu().numpy())
        texts.extend(text)
        golds.extend(y.cpu().numpy())
    epoch_metrics = metrics.get_metrics(preds, golds)
    if args.get_rationales:
        epoch_stat = {'loss' : float(np.round(np.mean(losses),3)), 'pred_loss': float(np.round(np.mean(pred_losses),3)), 'select_loss' : float(np.round(np.mean(select_losses),3))}
    else:
        epoch_stat = {'loss': float(np.mean(pred_losses))}
    epoch_stat.update(epoch_metrics)
    return(epoch_stat, step, rationales, texts)
