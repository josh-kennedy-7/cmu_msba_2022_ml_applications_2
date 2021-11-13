import random
import torch
from d2l import torch as d2l

import numpy as np
import pandas as pd

import os

def read_data_books():
    data_dir = "M:\\git\\ML2_2021\\data"
    names = ['user_id', 'isbn', 'rating']
    data = pd.read_csv(os.path.join(data_dir, 'book_ratings.csv'), ',', names=names,
                       engine='python', skiprows=1)

    data['isbn'] = data.isbn.astype('category')
    data['book_id'] = data['isbn'].cat.codes

    data['user_id'], _ = pd.factorize(data['user_id'])
    data['book_id'], _ = pd.factorize(data['book_id'])

    data.user_id = data.user_id
    data.book_id = data.book_id
    
    data.drop('isbn',axis=1,inplace=True)
    data = data[['user_id','book_id','rating']]
    
    data['rating'] = data['rating'].astype(float)

    num_users = data.user_id.unique().shape[0]
    num_items = data.book_id.unique().shape[0]
    return data, num_users, num_items


def tensor_data(data, n_user, n_item):  #@save
    users = torch.tensor(data['user_id'],dtype=torch.long)
    books = torch.tensor(data['book_id'],dtype=torch.long)
    rting = torch.tensor(data['rating'],dtype=torch.float32)                              

    X = torch.cat((users.reshape((-1,1)), books.reshape((-1,1))), 1)
    y = rting.reshape((-1,1))
    return X, y


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    # random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i +
                                                   batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]