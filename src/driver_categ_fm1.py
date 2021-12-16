import torch
from torch.utils.data import DataLoader

import os
import shutil
import numpy as np
import pandas as pd
from copy import deepcopy

from data_mgmt import BaseDataClass as bsd
from core.loops import train_loop, test_loop
from data_mgmt import ValidationBaseDataClass
from models.FactorioMachine import FactorizationMachineModel

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

""" A Note From Reed:

"""

def splitValidationByUser(ds_in):
    df_validate = ds_in.df_data.copy().groupby('reviewerID').last().reset_index()
    df_validate=df_validate.loc[ds_in.df_data.groupby('reviewerID').count().reset_index().reviewHash>1]
    ds_in.df_data=ds_in.df_data.set_index('reviewHash').drop(df_validate.reviewHash).reset_index()

    return ValidationBaseDataClass.ValidationDataClass(
                            df_validate, transform=ds_in.transform,
                            target_transform=ds_in.target_transform)

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

    ds_train = bsd.BaseDataClass(PATH_DATA)
    
    vect = TfidfVectorizer()
    stop = stopwords.words('english')
    stemmer = PorterStemmer()
    
    df = ds_train.df_data.copy()
    
    cols = ['summary','reviewText']
    df['Review_N_summary'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    
 #   df['Review_N_summary'] = df['Review_N_summary'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
 #   df['Review_N_summary'] = df['Review_N_summary'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    vect = vect.fit(df['Review_N_summary'])
    summary_xfmred = vect.transform(df['Review_N_summary'])
    df['Review_N_summary'] = summary_xfmred.tolil().rows
    dict_size = summary_xfmred.shape[1]
    
    ds_train.df_data = df.copy()
    
    # 90-10 split train-validate
    train_size = int(0.9 * len(ds_train))
    val_size = len(ds_train) - train_size

    df_train, df_val = torch.data.utils.random_split(ds_train, [train_size, val_size])

    print(len(df_train),len(df_val))


    tdl = DataLoader(df_train, batch_size=bsize, shuffle=True)
    vdl = DataLoader(df_val, batch_size=bsize, shuffle=True)




    #n_user = ds_train.df_data.uid.append(ds_valid.df_data.uid).unique().shape[0]
    # n_item = ds_train.df_data.pid.append(ds_valid.df_data.pid).unique().shape[0]
    model_dims = np.array([dict_size])

    model = FactorizationMachineModel(model_dims, ebdim)

    if tensorboard:
        tb.add_graph(model, next(iter(tdl))[0])

    model = model.to(device=device)


    loss = torch.nn.CrossEntropyLoss()

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
    #["adam_256_005_3e3_500_800_FULL", 256, 0.0005, 1e-5, 500, 'adam',800]
    torch.cuda.empty_cache()
    trials_and_tribulations = \
      [["adam_256_001_1e4_500_800_FULL", 256, 0.001, 1e-4, 500, 'adam',512+128]]
      #["pain2", 256, 0.01, 1e-3, 500, 'adam',64]]

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