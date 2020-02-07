import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# 1) load pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 2) tokenized input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP] "
tokenized_text = tokenizer.tokenize(text)

# 3) mask
mask_index = 8
tokenized_text[mask_index] = '[MASK]'

# 4) convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# 5) define sentence a and b indices associate ot 1st and 2nd sentences
segments_ids = [0. for i in range(7)] +[1. for i in range (7)]

# 6) convert inputs to pytorch tensor
token_tensor = torch.tensor([indexed_tokens]).long()
segments_tensors = torch.tensor([segments_ids]).long()

# load pre-trained model(Weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# token_tensor = token_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

with torch.no_grad():
    encoded_layers, _ = model(token_tensor, segments_tensors)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# token_tensor = token_tensor.to('cuda')
# segments_tensors = segments_tensors.to('cuda')
# model.to('cuda')

with torch.no_grad():
    predictions = model(token_tensor, segments_tensors)

predicted_index = torch.argmax(predictions[0, mask_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print(predicted_token)





