
import os
os.getcwd()
os.chdir('cmu_msba_2022_ml_applications_2/src') 
import data_mgmt.RecSysData as rsd
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch import autograd
import pandas as pd
import numpy as np
from data_mgmt.BaseDataClass import BaseDataClass

df = BaseDataClass._loadUpDf('/Users/joshkennedy/Documents/CMU/ML for Business Applications 2/','train','.json')

train_features = torch.tensor(np.array(rsd.RecSysData.recSysPreprocessing(df)[['uid','pid']]), dtype=torch.float32)
train_labels = torch.tensor(np.array(rsd.RecSysData.recSysPreprocessing(df)['rating']), dtype=torch.float32)
# os.getcwd()
# ppath = '/Users/joshkennedy/Documents/CMU/ML for Business Applications 2/train.json'
ppath='/Users/joshkennedy/GitKraken/cmu_msba_2022_ml_applications_2/data/'
# df.to_csv('cmu_msba_2022_ml_applications_2/data/clean_train_json.csv')


# The minimum argument RecSysData needs to create obj is the data path
tt = rsd.RecSysData(ppath)

train_loader = DataLoader(tt,batch_size=4000, shuffle = True)

next(iter(train_loader))

# tt.__dir__()

# tt.df_data[tt.df_data['pid'] == 4]

net = nn.Sequential(nn.Linear(2, 1), nn.ReLU())
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=10)

# train and testing
num_epochs = 3
for epoch in range(1, num_epochs + 1):
        for X, y in train_loader:
            l = loss(net(X), y)
            print(l)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(train_features), train_labels)
        # test_loss = loss(net(train_features), train_labels)
        # print('epoch %d, training loss: %f, testing loss: %f' % (epoch, l.mean(), test_loss.mean()))    
        print('epoch %d, training loss: %f' % (epoch, l.mean()))    

net(X)
X
net(train_features)

# compare to recsys1 (recys garbagio driver 1/2)
# Can I take this same structure and come up with a 3rd structure?
# Can I use it to do this same thing?
# need to increase the learning rate?
# transform the test data to be in a format that we can put into the trainer
# final output looks like a 3rd iteration on the rating system
# loss KPIs for the rating system
# switch to the recsys1 branch