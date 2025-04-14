from torch import nn
from transformers import AutoModel


class BERT(nn.Module):
    def __init__(self, args, encoding=None, bert_name='bert-base-uncased'):
        super(BERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        #self.embeddings = self.bert.embeddings
        self.embed_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.encoding = encoding

    def forward(self, input_id, att_mask, inputs_embeds=None):
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=att_mask)#,
        if self.encoding:#enc
            return outputs.last_hidden_state[:, 0]
        else:#gen
            return outputs.last_hidden_state
