

from datasets import load_dataset			# load hugging face datasets
from transformers import AutoTokenizer
import pickle
import torch

ds = load_dataset("presencesw/mednli")

# define training, test, validation split
train = load_dataset('presencesw/mednli', split = 'train')
test = load_dataset('presencesw/mednli', split = 'test')                    # save test sets as .csv
validation = load_dataset('presencesw/mednli', split = 'validation')


# label encoding & decoding maps
# entailment = 0, neutral = 1, contradiction = 2
encoding_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
decoding_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}


# bio-bert tokenizer trained on clinical notes
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

def tokenize(data):
    tokenized = tokenizer(
        data['sentence1'],
        data['sentence2'],
        truncation = True,
        padding = 'max_length',
        max_length = 128    # maximum 128 tokens
        )
    
    # create new col for label encoding
    tokenized['label'] = encoding_map[data['gold_label']]
    return tokenized

# tokenize texts & enode labels
train = train.map(tokenize)
validation = validation.map(tokenize)
test = test.map(tokenize)

# define models
# from hugging face
from transformers import AutoModelForSequenceClassification as amsc
# model_bert = amsc.from_pretrained('prajjwal1/bert-tiny', num_labels = 3)
model_bert = amsc.from_pretrained('bert', num_labels = 3)

from transformers import DistilBertForSequenceClassification
model_dbert = DistilBertForSequenceClassification.from_pretrained("distilbert")


model_bert.eval()

with torch.no_grad():
    preds = torch.argmax(model_bert(test).logits, dim=1)


