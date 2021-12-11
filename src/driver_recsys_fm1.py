import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import pandas as pd
from copy import deepcopy

from data_mgmt import RecSysData as rsd
from core.loops import train_loop, test_loop
from data_mgmt import ValidationBaseDataClass
from models.FactorioMachine import FactorizationMachineModel

""" A Note From Reed:

Behold, the travesty that is rec_sys_garbagio_driver_2.py...

This is attempt #2 at tuning the alfa + beta_u + beta_i model
using pytorch.

Look around line 21 (just below this) to be shocked at the
audacity that is RecSysGarbageNet. nn.Embedding is an
encoder class from PyTorch typically used for NLP.

I have hijacked it so that it essentially has two
vocabularies of one dimension per "word", except
each word is actually a user or item.

nn.Embedded takes in an index and then says
"Oh alright so this should depend on the index
yea alright that's cool" and then applies the
weights corresponding to that dictionary entry.

Runs a lot faster than the nn.Linear one (just a table lookup)
but tends to skyline around MSE=~3.2.

Now that I think about it setting all of the vocab lengths to 1
might mean I am accidentally setting the "hidden features" to size 1,
if we set this higher we may get better results.

Little too tired to think that through right now.

"""

def splitValidationByUser(ds_in):
    df_validate = ds_in.df_data.copy().groupby('reviewerID').last().reset_index()
    df_validate=df_validate.loc[ds_in.df_data.groupby('reviewerID').count().reset_index().reviewHash>1]
    ds_in.df_data=ds_in.df_data.set_index('reviewHash').drop(df_validate.reviewHash).reset_index()

    return ValidationBaseDataClass.ValidationDataClass(
                            df_validate, transform=ds_in.transform,
                            target_transform=ds_in.target_transform)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    torch.cuda.empty_cache()

    MODEL_NAME = "fm_testing"

    PATH_DATA  = os.path.abspath('data/')
    PATH_SAVE  = os.path.abspath('src/models/saved/')
    PATH_TBRD  = os.path.join(PATH_SAVE,MODEL_NAME)
    PATH_MWRT  = os.path.join(PATH_SAVE,MODEL_NAME+".pkl")

    if os.path.exists(PATH_TBRD):
        os.remove(PATH_TBRD)
    tb = SummaryWriter(os.path.join(PATH_SAVE,MODEL_NAME))
    MODEL_NAME += ".pkl"

    ds_train = rsd.RecSysData(PATH_DATA)
    #ds_train.df_data = ds_train.df_data.iloc[0:10000]
    ds_valid = splitValidationByUser(ds_train)

    tdl = DataLoader(ds_train, batch_size=128, shuffle=False)
    vdl = DataLoader(ds_valid, batch_size=128, shuffle=False)

    n_user = ds_train.df_data.uid.append(ds_valid.df_data.uid).unique().shape[0]
    n_item = ds_train.df_data.pid.append(ds_valid.df_data.pid).unique().shape[0]
    model_dims = np.array([n_user,n_item])

    model = FactorizationMachineModel(model_dims,64)
    tb.add_graph(model, next(iter(tdl))[0])

    model = model.to(device=device)

    learning_rate = 0.05

    loss = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=15.0, rho=0.9, eps=1e-06, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    mode='min', factor=0.666,
                    patience=5, threshold=0.02, threshold_mode='rel',
                    cooldown=5, min_lr=1e-6, eps=1e-08, verbose=True)

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer, method='encoder', in_device=device, board=tb, epoch=t)
        val_loss=test_loop(vdl, model, loss, method='encoder', in_device=device, board=tb, epoch=t)
        scheduler.step(val_loss)

        torch.save(deepcopy(model.state_dict()), os.path.join(PATH_SAVE,MODEL_NAME))

    print("Done!")

if __name__ == "__main__":
    main()