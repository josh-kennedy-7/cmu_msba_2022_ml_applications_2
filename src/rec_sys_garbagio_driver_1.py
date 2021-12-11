import data_mgmt.RecSysData as rsd
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
from core.loops import train_loop, test_loop
from copy import deepcopy
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

    ppath=r"C:\git\cmu_msba_2022_ml_applications_2\data"+"\\"
    ppath="//home/rster/sw/cmu_msba_2022_ml_applications_2/data/"
    MODEL_SAVE_PATH="//home/rster/sw/cmu_msba_2022_ml_applications_2/src/models/saved/"


    omfg = rsd.RecSysData(ppath, preprocess=overloadedPreProcess, transform=overloadedTransform)
    wtfbbq = splitValidationByUser(omfg)
    tdl = DataLoader(omfg, batch_size=4000, shuffle=False)
    vdl = DataLoader(wtfbbq, batch_size=4000, shuffle=False)

    n_user = omfg.df_data.uid.append(wtfbbq.df_data.uid).unique().shape[0]
    n_item = omfg.df_data.pid.append(wtfbbq.df_data.pid).unique().shape[0]

    learning_rate = 3e-2
    model = nn.Linear(n_user+n_item, 1).cuda()
    loss = torch.nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    mode='min', factor=0.666,
                    patience=3, threshold=0.0001, threshold_mode='abs',
                    cooldown=3, min_lr=1e-6, eps=1e-08, verbose=True)

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer, device='linmod')
        val_loss=test_loop(vdl, model, loss, device='linmod')

        scheduler.step(val_loss)
        torch.save(deepcopy(model.state_dict()), MODEL_SAVE_PATH+"jimlinear.pkl")

    print("Done!")

if __name__ == "__main__":
    main()