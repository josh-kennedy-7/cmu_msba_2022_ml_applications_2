from . import BaseDataClass
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np

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

        self.df_data, self.dict_size = self.preprocess(self.df_data)

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
        df_in=df_in.query("reviewHash!='R0'")
        df_in=df_in.iloc[0:10000]

        vect = TfidfVectorizer()
        stop = stopwords.words('english')
        stemmer = PorterStemmer()

        df = df_in.copy()

        cols = ['summary','reviewText']
        df['Review_N_summary'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        df['Review_N_summary'] = df['Review_N_summary'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
        df['Review_N_summary'] = df['Review_N_summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        vect = vect.fit(df['Review_N_summary'])
        summary_xfmred = vect.transform(df['Review_N_summary'])
        df['Review_N_summary'] = summary_xfmred.tolil().rows
        max_token_len = df.Review_N_summary.apply(lambda x: len(x)).max()

        df['Review_N_summary'] = df['Review_N_summary'].apply(lambda x: np.array(x))

        dict_size = summary_xfmred.shape[1]

        return (df, dict_size)

    @staticmethod
    def catPredXfrm(in_row):
        pad_size = 2000 - in_row.Review_N_summary.shape[0]
        return torch.nn.functional.pad(torch.tensor(in_row.Review_N_summary),(0,pad_size))

    @staticmethod
    def catPredTgtXfrm(in_row):
        return torch.tensor([in_row.categoryID], dtype=int)
