from transformers import AutoModel

def bert_embeddings(bert_name='bert-base-uncased'):
    embeddings_layer = AutoModel.from_pretrained(bert_name).embeddings.word_embeddings
    return(embeddings_layer)
