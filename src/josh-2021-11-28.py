import os
import cmu_msba_2022_ml_applications_2.src.data_mgmt.RecSysData as rsd
from torch.utils.data import DataLoader
import torch
import pandas as pd
from cmu_msba_2022_ml_applications_2.src.data_mgmt.BaseDataClass import BaseDataClass

df = BaseDataClass._loadUpDf('/Users/joshkennedy/Documents/CMU/ML for Business Applications 2/','train','.json')
df

ppath = '/Users/joshkennedy/Documents/CMU/ML for Business Applications 2/train.json'
ppath='cmu_msba_2022_ml_applications_2/data/'
df.to_csv('cmu_msba_2022_ml_applications_2/data/clean_train_json.csv')


# The minimum argument RecSysData needs to create obj is the data path
tt = rsd.RecSysData(ppath)

train_loader = DataLoader(tt,batch_size=4, shuffle = True)

next(iter(train_loader))

tt.__dir__()

tt.df_data[tt.df_data['pid'] == 4]