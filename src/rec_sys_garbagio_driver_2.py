import data_mgmt.RecSysData as rsd
from models.RecSysFlat import RecSysGarbageNetV2
from torch.utils.data import DataLoader
import torch
from core.loops import train_loop, test_loop
from copy import deepcopy

from torch.optim.swa_utils import AveragedModel, SWALR
from data_mgmt import ValidationBaseDataClass

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

    ppath=r"C:\git\cmu_msba_2022_ml_applications_2\data"+"\\"
    ppath="//home/rster/sw/cmu_msba_2022_ml_applications_2/data/"
    MODEL_SAVE_PATH="//home/rster/sw/cmu_msba_2022_ml_applications_2/src/models/saved/"

    omfg = rsd.RecSysData(ppath)
    wtfbbq = splitValidationByUser(omfg)
    tdl = DataLoader(omfg, batch_size=50000, shuffle=False)
    vdl = DataLoader(wtfbbq, batch_size=50000, shuffle=False)

    n_user = omfg.df_data.uid.append(wtfbbq.df_data.uid).unique().shape[0]
    n_item = omfg.df_data.pid.append(wtfbbq.df_data.pid).unique().shape[0]

    model = RecSysGarbageNetV2(n_user,n_item,16)
    swa_model = AveragedModel(model)

    model = model.cuda()
    swa_model = swa_model.cuda()

    loss = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-1)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=15.0, rho=0.9, eps=1e-06, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                    mode='min', factor=0.666,
                    patience=5, threshold=0.02, threshold_mode='rel',
                    cooldown=5, min_lr=1e-6, eps=1e-08, verbose=True)
    swa_scheduler = SWALR(optimizer, anneal_epochs=50, swa_lr=0.008)

    swa_start = 750
    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer, device='encoder')


        if t > swa_start:
            val_loss=test_loop(vdl, swa_model, loss, device='encoder')
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            val_loss=test_loop(vdl, model, loss, device='encoder')
            scheduler.step(val_loss)

        torch.save(deepcopy(model.state_dict()), MODEL_SAVE_PATH+"jimlad.pkl")
        torch.save(deepcopy(swa_model.state_dict()), MODEL_SAVE_PATH+"jimswa.pkl")

    print("Done!")

if __name__ == "__main__":
    main()