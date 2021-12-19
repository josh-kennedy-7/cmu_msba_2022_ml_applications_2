from . import BaseDataClass
import pandas as pd
import torch

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TODO -- REED -- ! ! ! ! ! HEY WE HAVEN'T EVEN STARTED THIS ONE YET! ! ! ! ! !
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PurchPredData(BaseDataClass.BaseDataClass):
    """PyTorch DataClass designed to tackle challenge #1 (Purchase Prediction)

    "Predict given a (user,item) pair from ‘pairs Purchase.txt’ whether the user purchased
    the item (really, whether it was one of the products they reviewed). Accuracy will be measured in terms
    of the categorization accuracy (fraction of correct predictions). The test set has been constructed such
    that exactly 50% of the pairs correspond to purchased items and the other 50% do not."

    Args:
        BaseDataClass (Python Class): This is the Base Class...
    """

    def __init__(self, root_dir, preprocess=None, transform=None, target_transform=None):
        super().__init__(root_dir)        
        
        if preprocess:
            self.preprocess = preprocess
        else:
            self.preprocess = self.purchPredPreprocessing
        
        self.df_data = self.preprocess(self.df_data)
        
        if transform:
            self.transform = transform
        else:
            self.transform = self.purchPredXfrm
            
        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = self.purchPredTgtXfrm

    @ staticmethod
    def purchPredPreprocessing(df_data):
        df_data['uid'], _ = pd.factorize(df_data['reviewerID'])
        df_data['pid'], _ = pd.factorize(df_data['itemID'])

        df_data = df_data[['reviewHash', 
                            'reviewerID',
                            'itemID',
                            'uid','pid','categoryID','qid']]

        return df_data


    def purchPredXfrm(self):
        return None

    def purchPredTgtXfrm(self):
        return None
