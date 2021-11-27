from data_mgmt.CatPredData import CatPredData
from transformers import BertTokenizerFast
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
    # eliminate blank row
    df_in=df_in[df_in.reviewHash!="R0"]
    df_in=df_in.iloc[0:10000]
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    tokenized = tokenizer(df_in.reviewText.tolist(), \
        padding=True, truncation=True, return_tensors="pt")
    
    return (df_in, tokenized['input_ids'])


def main():
    ppath="C:\\git\\cmu_msba_2022_ml_applications_2\\data\\"    
    cpd = CatPredData(ppath)    
    
if __name__ == "__main__":
    main()