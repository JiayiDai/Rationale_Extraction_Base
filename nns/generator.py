import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import nns.cnn as cnn
import nns.bilstm as bilstm
import nns.bert as bert
import nns.embedding as embedding

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.z_dim = 2#bernoulli distri for feature selection
        self.embedding_layer = embedding.bert_embeddings()
        self.embedding_layer.weight.requires_grad = False
        if args.model_form == 'cnn':
            self.cnn = cnn.CNN(args, max_pool_over_time = False)
            self.fc = nn.Linear((len(args.filters)* args.filter_num), self.z_dim)
        elif args.model_form == 'bilstm':
            self.bilstm = bilstm.BILSTM(args)
            self.fc = nn.Linear(args.bilstm_dim*2, self.z_dim)
        elif args.model_form == "bert":
            self.bert = bert.BERT(args, encoding=False)
            self.fc = nn.Linear(self.bert.embed_dim, self.z_dim)
        
        self.bn = nn.BatchNorm1d(200)#length
        #self.dropout = nn.Dropout(args.dropout)

    def gumbel_softmax(self, input, temperature, cuda):
        noise = torch.rand(input.size())
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = autograd.Variable(noise)
        if cuda:
            noise = noise.cuda()
        x = (input + noise) / temperature
        return F.softmax(x, dim=-1), F.log_softmax(x, dim=-1)

    def __z_forward(self, activ, model_form):
        '''
            Returns prob of each token being selected
        '''
        if model_form == "cnn":
            activ = activ.transpose(1,2)#batch*length*300
        else:
            pass
        logits = self.fc(activ)#batch*200*2 for bilstm
        probs, log_probs = self.gumbel_softmax(logits, self.args.gumbel_temprature, self.args.cuda)
        z = probs[:,:,1]
        return z, probs, log_probs
    
    def forward(self, x_indx, att_mask=None):
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on x
        '''
        x = self.embedding_layer(x_indx.squeeze(1))
        if self.args.cuda:
            x = x.cuda()
        if self.args.model_form == 'cnn':
            x = torch.transpose(x, 1, 2) # Switch X to (Batch, Embed, Length)
            activ = self.cnn(x)
            z, probs, log_probs = self.__z_forward(F.relu(activ), "cnn")
            
        elif self.args.model_form == "bilstm":
            activ = self.bilstm(x)
            z, probs, log_probs = self.__z_forward(F.relu(activ), "bilstm")
        elif self.args.model_form == "bert":
            activ = self.bert(x_indx, att_mask, inputs_embeds=x)
            z, probs, log_probs = self.__z_forward(F.relu(activ), "bert")
        else:
            raise NotImplementedError("Model form {} not yet supported for generator!".format(args.model_form))
        mask = self.sample(z)
        return mask, probs, log_probs

    def get_hard_mask(self, z):
        masked = torch.ge(z, 0.5).float()
        del z
        return masked

    def sample(self, z):
        '''
            Get mask from probablites at each token. Use gumbel
            softmax at train time, hard mask at test time
        '''
        mask = z
        if self.training:
            mask = z
        else:
            ## pointwise set <.5 to 0 >=.5 to 1
            mask = self.get_hard_mask(z)
        return mask

    def loss(self, mask):
        '''
            Compute the generator specific costs, i.e selection cost, continuity cost
        '''
        selection_cost = torch.mean(torch.sum(mask, dim=1))
        return selection_cost
