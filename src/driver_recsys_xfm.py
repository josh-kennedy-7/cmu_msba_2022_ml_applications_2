import torch
from torch.utils.data import DataLoader

import os
import shutil
import numpy as np
import pandas as pd
from copy import deepcopy

from data_mgmt import RecSysData as rsd
from core.loops import train_loop, test_loop
from data_mgmt import ValidationBaseDataClass
from models.ExtremeDeepFactorioMachine import ExtremeDeepFactorizationMachineModel as XFM

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

def recSysPreprocessing(df_data):
    df_data['uid'], _ = pd.factorize(df_data['reviewerID'])
    df_data['pid'], _ = pd.factorize(df_data['itemID'])
    df_data['cid'], _ = pd.factorize(df_data['categoryID'])

    df_data = df_data[['reviewHash', 'reviewerID',
                        'unixReviewTime', 'itemID','categoryID',
                        'rating', 'uid','pid','cid']]

    return df_data

def recSysXfrm(in_row):
    return torch.tensor([in_row.uid,in_row.pid,in_row.cid],dtype=torch.int)

def adam_driver(MODEL_NAME      = "default_model",
                bsize           = 128,
                learning_rate   = 0.05,
                decay           = 1e-4,
                epochs          = 50,
                loss_fn_name    = 'adam',
                model_params    = (32,(256,256),(48, 48, 48)),
                tensorboard     = True):

    mp_embeddim,mp_mlpdims,mp_cinsz = model_params[0], model_params[1], model_params[2]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    torch.cuda.empty_cache()

    PATH_DATA  = os.path.abspath('data/')
    PATH_SAVE  = os.path.abspath('src/models/saved/')
    PATH_TBRD  = os.path.join(PATH_SAVE,MODEL_NAME)
    PATH_MWRT  = os.path.join(PATH_SAVE,MODEL_NAME+".pkl")

    if os.path.exists(PATH_TBRD):
        shutil.rmtree(PATH_TBRD)

    tb = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(PATH_SAVE,MODEL_NAME))
    MODEL_NAME += ".pkl"

    ds_train = rsd.RecSysData(PATH_DATA, preprocess=recSysPreprocessing, transform=recSysXfrm)
    n_user = ds_train.df_data.copy().uid.unique().shape[0]
    n_item = ds_train.df_data.copy().pid.unique().shape[0]
    n_cats = ds_train.df_data.copy().cid.unique().shape[0]

    #ds_train.df_data = ds_train.df_data.iloc[np.random.randint(0,high=80000,size=10000)]
    ds_valid = splitValidationByUser(ds_train)

    tdl = DataLoader(ds_train, batch_size=bsize, shuffle=True)
    vdl = DataLoader(ds_valid, batch_size=bsize, shuffle=True)

    model_dims = np.array([n_user,n_item,n_cats])

    model = XFM(field_dims = model_dims,
                embed_dim = mp_embeddim,
                mlp_dims  = mp_mlpdims,
                dropout = 0.33,
                cross_layer_sizes = mp_cinsz,
                split_half=False)

    #if tensorboard:
        #tb.add_graph(model, next(iter(tdl))[0])

    model = model.to(device=device)


    loss = torch.nn.MSELoss(reduction='mean')

    if loss_fn_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    elif loss_fn_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    elif loss_fn_name == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=decay)


    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                 mode='min', factor=0.666,
    #                 patience=10, threshold=0.01, threshold_mode='rel',
    #                 cooldown=10, min_lr=1e-6, eps=1e-08, verbose=True)

    tri_step_size = round(epochs/3)
    tri_step_size = 15

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                    learning_rate, 20*learning_rate, step_size_up=tri_step_size,
                    mode='triangular2',
                    cycle_momentum=False, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer, method='encoder', in_device=device, board=tb, epoch=t)
        val_loss=test_loop(vdl, model, loss, method='encoder', in_device=device, board=tb, epoch=t)

        if issubclass(scheduler.__class__, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if tensorboard:
            tb.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], t)

        torch.save(deepcopy(model.state_dict()), os.path.join(PATH_SAVE,MODEL_NAME))

    print("Done!")

import time

def main():
    torch.cuda.empty_cache()
    trials_and_tribulations = \
      [ ["depth_516_FULL", 64, 0.0005, 1e-16, 500, 'adamw', (512, 10*(8,8),2*(50,50))] ]

# ["depth_501", 512, 0.0005, 1e-15, 25, 'adamw', (512, 8*(8,8),1*(96,96,96))]


    for trial in trials_and_tribulations:
        adam_driver(MODEL_NAME      = trial[0],
                    bsize           = trial[1],
                    learning_rate   = trial[2],
                    decay           = trial[3],
                    epochs          = trial[4],
                    loss_fn_name    = trial[5],
                    model_params    = trial[6])

        time.sleep(3)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()