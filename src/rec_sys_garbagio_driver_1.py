import data_mgmt.RecSysData as rsd
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm


""" A Note From Reed:

Behold, the travesty that is rec_sys_garbagio_driver_1.py...

This is attempt #1 at tuning the alfa + beta_u + beta_i model
using pytorch.

The data transformer is overloaded so the the pytorch nn gets a
flat tensor with columns first corresponding to one-hot userID
and then one-hot itemID. If it is a user/item, then those cols
will be one, otherwise they will be zero.

nn.Linear doesn't like integers. So I've made the one hot vector
floats. Alpha (the bias) is actually already encapsulated in the
nn.Linear module and doesn't need to be explicitly defined.

MSE starts bottoming out around 1.18, although I'm not so sure
I've done the scheduling right.


Console output from when I ran it is below:

Epoch 1
-------------------------------
100%|| 800/800 [03:18<00:00,  4.29it/s, current_loss=1.2]
Epoch Done, max loss:19.866558074951172, min loss: 0.8208678960800171, final loss: 1.1964223384857178
Adjusting learning rate of group 0 to 6.0000e-03.
Epoch 2
-------------------------------
100%|| 800/800 [03:10<00:00,  4.12it/s, current_loss=1.19]
Epoch Done, max loss:1.6257232427597046, min loss: 0.8116060495376587, final loss: 1.188079595565796
Adjusting learning rate of group 0 to 1.2000e-03.
Epoch 3
-------------------------------
100%|| 800/800 [03:08<00:00,  4.18it/s, current_loss=1.19]
Epoch Done, max loss:1.629172682762146, min loss: 0.8022348284721375, final loss: 1.1870187520980835
Adjusting learning rate of group 0 to 2.4000e-04.
Epoch 4
-------------------------------
100%|| 800/800 [03:18<00:00,  4.12it/s, current_loss=1.19]
Epoch Done, max loss:1.6303538084030151, min loss: 0.802473783493042, final loss: 1.186949610710144
Adjusting learning rate of group 0 to 4.8000e-05.
Epoch 5
-------------------------------
100%|| 800/800 [03:09<00:00,  4.15it/s, current_loss=1.19]
Epoch Done, max loss:1.6300902366638184, min loss: 0.8026332855224609, final loss: 1.1869014501571655
Adjusting learning rate of group 0 to 9.6000e-06.
Epoch 6
-------------------------------
100%|| 800/800 [03:19<00:00,  4.08it/s, current_loss=1.19]
Epoch Done, max loss:1.6300303936004639, min loss: 0.8025795817375183, final loss: 1.186890959739685
Adjusting learning rate of group 0 to 1.9200e-06.
Epoch 7
-------------------------------
100%|| 800/800 [03:23<00:00,  3.53it/s, current_loss=1.19]
Epoch Done, max loss:1.6300185918807983, min loss: 0.8025672435760498, final loss: 1.1868890523910522
Adjusting learning rate of group 0 to 3.8400e-07.
Epoch 8
-------------------------------
100%|| 800/800 [03:31<00:00,  3.89it/s, current_loss=1.19]
Epoch Done, max loss:1.6300171613693237, min loss: 0.8025640845298767, final loss: 1.1868888139724731
Adjusting learning rate of group 0 to 7.6800e-08.
Epoch 9
-------------------------------
100%|| 800/800 [03:26<00:00,  3.84it/s, current_loss=1.19]
Epoch Done, max loss:1.6300169229507446, min loss: 0.8025626540184021, final loss: 1.1868888139724731
Adjusting learning rate of group 0 to 1.5360e-08.
Epoch 10
-------------------------------
100%|| 800/800 [03:37<00:00,  4.07it/s, current_loss=1.19]
Epoch Done, max loss:1.6300169229507446, min loss: 0.8025626540184021, final loss: 1.1868888139724731

"""


def overloadedPreProcess(df_data):
    df_data['uid'], _ = pd.factorize(df_data['reviewerID'])
    df_data['pid'], _ = pd.factorize(df_data['itemID'])
    
    df_data['n_user'] = df_data.uid.unique().shape[0]
    df_data['n_item'] =df_data.pid.unique().shape[0]
    
    df_data = df_data[['reviewHash', 'reviewerID', 
                    'unixReviewTime', 'itemID', 
                    'rating', 'uid','pid','summaryCharacterLength',
                    'n_user','n_item']]
    
    return df_data

def overloadedTransform(in_row):    
    tt_users = torch.zeros(in_row.n_user,dtype=torch.float)
    tt_items = torch.zeros(in_row.n_item,dtype=torch.float)
    
    tt_users[in_row.uid] = 1
    tt_items[in_row.pid] = 1
    
    return torch.cat((tt_users,tt_items),dim=0)

    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    max_loss = 0.0
    min_loss = 9e9
    
    with tqdm(total=len(dataloader),leave=False) as t:
            
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X).flatten()
            loss = loss_fn(pred, y)
            
            if loss.item() > max_loss:
                max_loss = loss.item()
                
            if loss.item() < min_loss:
                min_loss = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t.set_postfix(current_loss=loss.item(), refresh=False)
            t.update()
        
        print(f"Epoch Done, max loss:{max_loss}, min loss: {min_loss}, final loss: {loss.item()}")
    
def main():
    ppath=r"C:\git\cmu_msba_2022_ml_applications_2\data"+"\\"
    omfg = rsd.RecSysData(ppath, preprocess=overloadedPreProcess, transform=overloadedTransform)
    tdl = DataLoader(omfg, batch_size=250)

    n_user = omfg.df_data.n_user[0]
    n_item = omfg.df_data.n_item[0]

    learning_rate = 3e-2
    model = nn.Linear(n_user+n_item, 1)  #RecSysGarbageNet2(n_user,n_item)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.2, verbose=True)
    
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer)
        #test_loop(test_dataloader, model, loss_fn)
        
        scheduler.step()
    print("Done!")
    
if __name__ == "__main__":
    main()