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
from models.FactorioMachine import FactorizationMachineModel

""" A Note From Reed:

"""
"""
START OF FUNCTION ALTERATIONS
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

    df_data = df_data[['reviewHash', 
                        'reviewerID',
                        'itemID',
                        'uid','pid','categoryID','qid',
                        'rating']]

    return df_data


def recSysXfrm(in_row):
    return torch.tensor([in_row.uid,in_row.pid,in_row.categoryID,in_row.qid],dtype=torch.int)
"""
END OF FUNCTION ALTERATIONS
"""


"""
START OF DRIVER SCRIPT
"""
def adam_driver(MODEL_NAME      = "default_model",
                bsize           = 128,
                learning_rate   = 0.05,
                decay           = 1e-4,
                epochs          = 50,
                loss_fn_name    = 'adam',
                ebdim           = 64,
                tensorboard     = True):

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

    # TRAINING DATASET
    ds_train = rsd.RecSysData(PATH_DATA, preprocess=recSysPreprocessing, transform=recSysXfrm)
    
    # get the sizes of the various indices
    n_users         = ds_train.df_data.uid.unique().shape[0]
    n_items         = ds_train.df_data.pid.unique().shape[0]
    n_categories    = ds_train.df_data.categoryID.unique().shape[0]
    n_deciles       = ds_train.df_data.qid.unique().shape[0]
    
    # VALIDATION DATASET
    ds_valid = splitValidationByUser(ds_train)

    tdl = DataLoader(ds_train, batch_size=bsize, shuffle=True)
    vdl = DataLoader(ds_valid, batch_size=bsize, shuffle=True)

    # store the index dimensions for the factorization machine
    model_dims = np.array([n_users,n_items,n_categories,n_deciles])

    model = FactorizationMachineModel(model_dims, ebdim)

    if tensorboard:
        tb.add_graph(model, next(iter(tdl))[0])

    model = model.to(device=device)


    loss = torch.nn.MSELoss(reduction='mean')

    if loss_fn_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    elif loss_fn_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    elif loss_fn_name == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=decay)

    tri_step_size = round(epochs/3)
    tri_step_size = 10

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                    learning_rate, 10*learning_rate, step_size_up=tri_step_size,
                    cycle_momentum=False, verbose=True)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer, method='encoder', in_device=device, board=tb, epoch=t)
        val_loss=test_loop(vdl, model, loss, method='encoder', in_device=device, board=tb, epoch=t)
        scheduler.step() #val_loss

        if tensorboard:
            tb.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], t)

        torch.save(deepcopy(model.state_dict()), os.path.join(PATH_SAVE,MODEL_NAME))

    print("Done!")

import time

def main():
    torch.cuda.empty_cache()
    trials_and_tribulations = \
      [["added_decile_feature_1", 256, 0.01, 1e-8, 25, 'adam',32]]

    for trial in trials_and_tribulations:
        adam_driver(MODEL_NAME      = trial[0],
                    bsize           = trial[1],
                    learning_rate   = trial[2],
                    decay           = trial[3],
                    epochs          = trial[4],
                    loss_fn_name    = trial[5],
                    ebdim           = trial[6])

        time.sleep(3)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()