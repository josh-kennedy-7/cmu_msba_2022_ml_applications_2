from . import BaseDataClass
import pandas as pd
import torch

class RecSysData(BaseDataClass.BaseDataClass):
    """PyTorch DataClass designed to tackle challenge #3 (Rating Prediction)

    "Predict people’s star ratings as accurately as possible, for those
    (user,item) pairs in ‘pairs Rating.txt’. Accuracy will be measured
    in terms of the mean-squared error (MSE)."

    Args:
        BaseDataClass (Python Class): This is the Base Class...
    """

    def __init__(self, root_dir, preprocess=None, transform=None, target_transform=None):
        """automatically invoked at time of object creation.

        ALRIGHT Y'ALL NOW LISTEN UP.

        You have 3 optional arguments to this function, they will be described below.

        This RecSys dataset will automatically whittle down the imported dataframe
        to only keep information relevant to the whole Alfa+Beta_U+Beta_I format.

        If you want to boil anything else in, DO NOT OVERWRITE the functions in this
        class! You can write functions elsewhere (like in a jupyter notebook) and over
        load it to do what you want!!!

        Args:
            root_dir (str): file path to the root directory of the data
            preprocess (FUNCTION, optional): class function to scrub at init. Defaults to None.
            transform (FUNCTION, optional): defines feature transformation in __getitem__. Defaults to None.
            target_transform (FUNCTION, optional): defines label transformation in __getitem__. Defaults to None.
        """
        super().__init__(root_dir)

        if preprocess:
            self.preprocess = preprocess
        else:
            self.preprocess = self.recSysPreprocessing

        self.df_data = self.preprocess(self.df_data)

        if transform:
            self.transform = transform
        else:
            self.transform = self.recSysXfrm

        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = self.recSysTgtXfrm


    @staticmethod
    def recSysPreprocessing(df_data):
        df_data['uid'], _ = pd.factorize(df_data['reviewerID'])
        df_data['pid'], _ = pd.factorize(df_data['itemID'])

        df_data = df_data[['reviewHash', 'reviewerID',
                            'unixReviewTime', 'itemID',
                            'rating', 'uid','pid']]

        return df_data

    @staticmethod
    def recSysXfrm(in_row):
        return torch.tensor([in_row.uid,in_row.pid],dtype=torch.int)

    @staticmethod
    def recSysTgtXfrm(in_row):
        return torch.tensor(in_row.rating, dtype=torch.float32)