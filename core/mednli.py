# packages & dependencies

from datasets import load_dataset			# load hugging face datasets
from collections import Counter
import pandas as pd
import numpy as np
import torch

import tokenizers                           # tokenizers from hugging face
import transformers                         # transformers from hugging face

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification as amsc
from transformers import TrainingArguments, Trainer

from sklearn.metrics import accuracy_score, f1_score, classification_report
# ************************************************************************

# load dataset from hugging face
# [huggingface data page](https://huggingface.co/datasets/presencesw/mednli) 

ds = load_dataset("presencesw/mednli")


# define training, test, validation split
train = load_dataset('presencesw/mednli', split = 'train')
test = load_dataset('presencesw/mednli', split = 'test')                    # save test sets as .csv
validation = load_dataset('presencesw/mednli', split = 'validation')


# pre-processing
# ************************************************************************

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



# define model

model = amsc.from_pretrained('prajjwal1/bert-tiny', num_labels = 3)

model = amsc.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', num_labels = 3)


# model training
training_params = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size = 64,
    # per_device_eval_batch_size = 64,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted'),  # or 'macro'
    }

trainer = Trainer(
    model = model,
    args = training_params,
    train_dataset = train,
    eval_dataset = validation,  
    compute_metrics = compute_metrics,
)

# train.set_format(
#     type='torch',
#     columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
# )

# validation.set_format(
#     type='torch',
#     columns=['input_ids', 'attention_mask', 'token_type_ids', 'label']
# )


# check gpu acceleration availability
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else: 
    device = 'cpu'


print(device)

model = model.to(device)


trainer.train()


# make predictions with tuned model
pred = trainer.predict(validation)

# convert logits to class
ypred = np.argmax(pred.predictions, axis = 1)

ypred

# convert torch tensor to list
yval = validation['label']

# create classification report
classification_report(yval, ypred)



from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# model training
training_params = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size = 64,
    # per_device_eval_batch_size = 64,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=1
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='weighted'),  # or 'macro'
    }

trainer = Trainer(
    model = model,
    args = training_params,
    train_dataset = train,
    eval_dataset = validation,  
    compute_metrics = compute_metrics,
)

model = model.to(device)

trainer.train()


# make predictions with tuned model
pred = trainer.predict(validation)

# convert logits to class
ypred = np.argmax(pred.predictions, axis = 1)

ypred

# convert torch tensor to list
yval = validation['label']

# create classification report
classification_report(yval, ypred)

