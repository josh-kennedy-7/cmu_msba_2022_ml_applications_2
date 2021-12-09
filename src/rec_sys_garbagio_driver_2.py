import data_mgmt.RecSysData as rsd
from models.RecSysFlat import RecSysGarbageNetV2
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from core.loops import train_loop, test_loop
from copy import deepcopy

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
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ppath=r"C:\git\cmu_msba_2022_ml_applications_2\data"+"\\"
    ppath="//home/rster/sw/cmu_msba_2022_ml_applications_2/data/"
    MODEL_SAVE_PATH="//home/rster/sw/cmu_msba_2022_ml_applications_2/src/models/saved/jimlad.pkl"

    omfg = rsd.RecSysData(ppath)
    #wtfbbq = omfg.splitValidation(preshuffle=True)
    tdl = DataLoader(omfg, batch_size=50000, shuffle=True)
    #vdl = DataLoader(wtfbbq, batch_size=200000, shuffle=True)

    n_user = omfg.df_data.uid.unique().shape[0] #+ wtfbbq.df_data.uid.unique().shape[0]
    n_item = omfg.df_data.pid.unique().shape[0] #+ wtfbbq.df_data.pid.unique().shape[0]

    learning_rate = 1.0
    model = RecSysGarbageNetV2(n_user,n_item,300)
    model = model.cuda()

    loss = torch.nn.L1Loss(reduction='sum')
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=10.0, rho=0.9, eps=1e-06, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
    #     patience=3, threshold=0.0001, threshold_mode='rel',
    #     cooldown=0, min_lr=0, eps=1e-08, verbose=True)

    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer, device)
        torch.save(deepcopy(model.state_dict()), MODEL_SAVE_PATH)
        # val_loss=test_loop(vdl, model, loss, device)

        #scheduler.step(val_loss.to('cpu'))
        scheduler.step()
    print("Done!")

if __name__ == "__main__":
    main()