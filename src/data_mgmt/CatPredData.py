from . import BaseDataClass
import pandas as pd
import torch

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TODO -- REED -- ! ! ! ! ! HEY WE HAVEN'T EVEN STARTED THIS ONE YET! ! ! ! ! !
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class CatPredData(BaseDataClass.BaseDataClass):
    """PyTorch DataClass designed to tackle challenge #1 (Purchase Prediction)

    "Predict given a (user,item) pair from ‘pairs Purchase.txt’ whether the user purchased
    the item (really, whether it was one of the products they reviewed). Accuracy will be measured in terms
    of the categorization accuracy (fraction of correct predictions). The test set has been constructed such
    that exactly 50% of the pairs correspond to purchased items and the other 50% do not."

    Args:
        BaseDataClass (Python Class): This is the Base Class...
    """

    def __init__(self, root_dir, preprocess=None, transform=None,
                 target_transform=None):
        super().__init__(root_dir)        
        
        if preprocess:
            self.preprocess = preprocess
        else:
            self.preprocess = self.catPredPreprocessing
        
        self.df_data, self.tf_tokenids = self.preprocess(self.df_data)
        
        if transform:
            self.transform = transform
        else:
            self.transform = self.catPredXfrm
            
        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = self.catPredTgtXfrm

    @staticmethod
    def catPredPreprocessing(df_in):
        # eliminate blank row
        # TODO -- REED -- this should probably be in the base class
        df_in=df_in[df_in.reviewHash!="R0"]
        
        # TODO -- REED -- enable full data set
        # for now only look at 20,000 examples to prevent
        # computer melting issues.
        df_in=df_in.iloc[0:20000]
        
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        tokenized = tokenizer(df_in.reviewText.tolist(), \
            padding=True, truncation=True, return_tensors="pt")
    
        return (df_in, tokenized['input_ids'])

    @staticmethod
    # TODO -- REED -- AUUUGH IT'S A STATIC METHOD THAT THINKS IT'S
    # GOING TO GET A DATAFRAME ROW IN THE BASE CLASS I'VE CODED
    # MYSELF INTO A LITTLE BOX
    # ok deep inhale, I'll fix this one later...
    # Anyways -> This one doesn't work for now
    # Check out catsys_testing.py
    def catPredXfrm(in_row):
        return self.tf_tokenids[:,in_row]

    @staticmethod
    def catPredTgtXfrm():
        return None
