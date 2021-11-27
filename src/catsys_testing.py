from data_mgmt.CatPredData import CatPredData
from transformers import BertTokenizerFast

from torch.utils.data import DataLoader
import torch
import pandas as pd

"""
Yeah so you're gonna want to read this:
https://huggingface.co/transformers/preprocessing.html

And this:
https://huggingface.co/transformers/main_classes/tokenizer.html

And this:
https://huggingface.co/transformers/model_doc/bert.html
"""


def bertPreProcess(df_in):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def main():
    ppath="C:\\git\\cmu_msba_2022_ml_applications_2\\data\\"    
    cpd = CatPredData(ppath)
    
    cpd.df_data=cpd.df_data[cpd.df_data.reviewHash!="R0"]
    
    print(cpd.df_data.head())
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    tokenized = tokenizer(cpd.df_data.reviewText.tolist(), \
        padding=True, truncation=True, return_tensors="pt")    
    print(tokenized['input_ids'].shape)
    print('\n\n')
    
if __name__ == "__main__":
    main()