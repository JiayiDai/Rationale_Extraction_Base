import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import nns.cnn as cnn
import nns.bilstm as bilstm
import nns.bert as bert
import pdb
import numpy as np
import nns.embedding as embedding

class Encoder(nn.Module):
    def __init__(self, args, embeddings=None):
        super(Encoder, self).__init__()
        self.args = args
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time=True)
            self.fc = nn.Linear( len(args.filters)*args.filter_num,  args.hidden_dim)
            self.hidden = nn.Linear(args.hidden_dim, args.num_class)
        elif args.model_form == "bilstm":
            self.bilstm = bilstm.BILSTM(args, bilstm_dim=args.bilstm_dim, encoding=True)
            self.fc = nn.Linear(args.bilstm_dim*4,  args.hidden_dim)
            self.hidden = nn.Linear(args.hidden_dim, args.num_class)
        elif args.model_form == "bert":
            self.bert = bert.BERT(args, encoding=True)
            self.fc = nn.Linear(self.bert.embed_dim, args.hidden_dim)
            self.hidden = nn.Linear(args.hidden_dim, args.num_class)
        else:
            raise NotImplementedError("Model form {} not yet supported for encoder!".format(args.model_form))
        self.embedding_layer = embedding.bert_embeddings()
        self.embedding_layer.weight.requires_grad = True
        self.dropout = nn.Dropout(args.dropout)
        self.bn = nn.BatchNorm1d(args.hidden_dim)

    def forward(self, x_indx, att_mask=None, mask=None):
        '''
            x_indx:  batch of word indices
            mask: Mask to apply over embeddings for tao ratioanles
        '''
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda:
            x = x.cuda()
            if mask != None:
                mask = mask.cuda()
                
        if not mask is None:
            x = x * mask.unsqueeze(-1)

        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            hidden = self.cnn(x)
            hidden = F.relu( self.fc(hidden) )
            hidden = self.dropout(hidden)
            logit = self.hidden(hidden)
            
        elif self.args.model_form == 'bilstm':
            hidden = self.bilstm(x)
            hidden = F.relu(self.fc(hidden))
            hidden = self.dropout(hidden)
            logit = self.hidden(hidden)
        elif self.args.model_form == 'bert':
            hidden = self.bert(x_indx, att_mask, inputs_embeds=x)
            hidden = F.relu(self.fc(hidden))
            hidden = self.dropout(hidden)
            logit = self.hidden(hidden)
        else:
            raise Exception("Model form {} not yet supported for encoder!".format(self.args.model_form))
        return logit
