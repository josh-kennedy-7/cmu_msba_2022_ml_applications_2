import torch
from torch import nn
from torch.utils.data import DataLoader

import os
import pandas as pd
from copy import deepcopy

from data_mgmt import RecSysData as rsd
from core.loops import train_loop, test_loop
from data_mgmt import ValidationBaseDataClass

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

def splitValidationByUser(ds_in):
    df_validate = ds_in.df_data.copy().groupby('reviewerID').last().reset_index()
    df_validate=df_validate.loc[ds_in.df_data.groupby('reviewerID').count().reset_index().reviewHash>1]
    ds_in.df_data=ds_in.df_data.set_index('reviewHash').drop(df_validate.reviewHash).reset_index()

    return ValidationBaseDataClass.ValidationDataClass(
                            df_validate, transform=ds_in.transform,
                            target_transform=ds_in.target_transform)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    PATH_DATA  = os.path.abspath('data/')
    PATH_SAVE  = os.path.abspath('src/models/saved/')
    MODEL_NAME = "garbagio1" + ".pkl"

    ds_train = rsd.RecSysData(PATH_DATA, preprocess=overloadedPreProcess, transform=overloadedTransform)
    ds_train.df_data = ds_train.df_data.iloc[0:10000]
    ds_valid = splitValidationByUser(ds_train)

    tdl = DataLoader(ds_train, batch_size=1000, shuffle=False)
    vdl = DataLoader(ds_valid, batch_size=1000, shuffle=False)

    n_user = ds_train.df_data.uid.append(ds_valid.df_data.uid).unique().shape[0]
    n_item = ds_train.df_data.pid.append(ds_valid.df_data.pid).unique().shape[0]

    learning_rate = 3e-2
    model = nn.Linear(n_user+n_item, 1).to(device=device)
    loss = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    mode='min', factor=0.666,
                    patience=3, threshold=0.0001, threshold_mode='abs',
                    cooldown=3, min_lr=1e-6, eps=1e-08, verbose=True)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer, method='linmod', in_device=device)
        val_loss=test_loop(vdl, model, loss, method='linmod', in_device=device)

        scheduler.step(val_loss)
        torch.save(deepcopy(model.state_dict()), os.path.join(PATH_SAVE,MODEL_NAME))

    print("Done!")

if __name__ == "__main__":
    main()