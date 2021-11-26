import data_mgmt.RecSysData as rsd
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm


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


class RecSysGarbageNet(nn.Module):
    def __init__(self, n_user, n_item):
        super(RecSysGarbageNet, self).__init__()
        self.alfa = nn.Parameter(torch.zeros([1,1]),requires_grad=True)
        self.beta_u = nn.Embedding(n_user, 1)
        self.beta_i = nn.Embedding(n_item, 1)    

    def forward(self,x):
        x = self.beta_u(x[:,0]) + self.beta_i(x[:,1])
        out = x + self.alfa
        return out
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    max_loss = 0.0
    min_loss = 9e9
    
    with tqdm(total=len(dataloader),leave=False) as t:
            
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X.type(torch.long)).flatten()
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
    omfg = rsd.RecSysData(ppath)
    tdl = DataLoader(omfg, batch_size=4000, shuffle=True)

    n_user = omfg.df_data.uid.unique().shape[0]
    n_item = omfg.df_data.pid.unique().shape[0]

    learning_rate = 1e-1
    model = RecSysGarbageNet(n_user,n_item)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=10.0, rho=0.9, eps=1e-06, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.92, verbose=True)
    
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(tdl, model, loss, optimizer)
        #test_loop(test_dataloader, model, loss_fn)
        
        scheduler.step()
    print("Done!")
    
if __name__ == "__main__":
    main()