import torch  , numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=3)
def tokenize_function(examples):
    return tokenizer(examples["clean_tweet"], padding="max_length", truncation=True, is_split_into_words=False)

def preparing_data():
    train = pd.read_csv('ML_Pytorch_repo\\fine_tunning\\amharic_sentiment_analysis\\data\\train.csv')
    test = pd.read_csv("ML_Pytorch_repo\\fine_tunning\\amharic_sentiment_analysis\\data\\test.csv")
    val = pd.read_csv("ML_Pytorch_repo\\fine_tunning\\amharic_sentiment_analysis\\data\\dev.csv")
    train["clean_tweet"] = train["clean_tweet"].astype(str)
    test["clean_tweet"] = test["clean_tweet"].astype(str)

    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    input_ids = train_dataset[0]["input_ids"]

    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_dataset , test_dataset

