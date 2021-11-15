from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import torch


class TwitterCovidDataset(Dataset):
    """This is a Dataset object from pytorch for our covid dataset

    Args:
        csv_file (location): path to the data
    """
    def __init__(self, csv_file='Data/Corona_NLP_train.csv'):
        df_in = pd.read_csv(csv_file, encoding='latin-1')
        df_in = self.data_transformer(df_in)
        
        self.df_tweets = df_in
        
    
    def data_transformer(self, df):
        '''
        Change multiclass to binary class: positive or negative tweets only
        Apply sklearn encoding on Sentiment column
        
        Param: Dataframe to transform
        Returns: Transformed dataframe
        '''
        df['Sentiment'] = df['Sentiment'].map({'Positive':'Positive', 'Extremely Positive':'Positive', 
                                            'Negative':'Negative', 'Extremely Negative':'Negative',
                                            'Neutral':'Positive'
                                            })
        df = df.drop(['UserName','ScreenName','Location','TweetAt'], axis=1)
        
        # Encode sentiment values
        df_le = LabelEncoder().fit(df['Sentiment'])
        df['encoded_sentiment'] = df_le.transform(df['Sentiment'])
        
        return df
    
    def __len__(self):
        return len(self.df_tweets)
    
    def __getitem__(self, idx):
        """returns the sample
        
        our sample is a tuple of the tweet and then the sentiment score
        
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        X_out = self.df_tweets['OriginalTweet'].iloc[idx]
        y_out = self.df_tweets['encoded_sentiment'].iloc[idx]
            
        sample = (y_out, X_out)
            
        return sample
    
    
# Just for testing, shouldn't run under other circumstances
if __name__ == "__main__":
    transformed_dataset = TwitterCovidDataset("Data/Corona_NLP_train.csv")
    
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == 3:
            print(sample_batched[0])
            print(sample_batched[1])
            break