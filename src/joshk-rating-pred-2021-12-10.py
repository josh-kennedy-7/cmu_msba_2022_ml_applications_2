import os
os.getcwd()
os.chdir('cmu_msba_2022_ml_applications_2/src')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import tarfile
from tqdm import tqdm
import json
import numpy as np
import time
from copy import deepcopy
from data_mgmt.BaseDataClass import BaseDataClass

# try this: SVDpp

from surprise import SVD
from surprise import SVDpp
from surprise import accuracy
from surprise import Reader
from surprise import Dataset
from surprise import BaselineOnly

rawdf = BaseDataClass._loadUpDf('/Users/joshkennedy/Documents/CMU/ML for Business Applications 2/','train','.json')
rawdf['date'] = rawdf['reviewTime']
df = rawdf.copy()
df.to_csv('clean_df.csv')

SAMPLING_RATE = 5/5
user_id_unique = df['reviewerID'].unique()
user_id_sample = pd.DataFrame(user_id_unique, columns=['unique_user_id']) \
                    .sample(frac= SAMPLING_RATE, replace=False, random_state=1)
ratings_sample = df.merge(user_id_sample, left_on='reviewerID', right_on='unique_user_id') \
                    .drop(['unique_user_id'], axis=1)

all_users = pd.DataFrame(df.groupby('reviewerID').count()['rating']).reset_index()
all_users.columns = ['reviewerID','num_ratings']
users_more_than_2_purchases = all_users[all_users['num_ratings'] > 2].copy()
users_more_than_5_purchases = all_users[all_users['num_ratings'] > 5].copy()
avg_item_ratings_dict = dict(df.groupby('itemID').mean()['rating'])
num_item_ratings_dict = dict(df.groupby('itemID').count()['rating'])
num_user_ratings_dict = dict(df.groupby('reviewerID').count()['rating'])

# most recent review is testset, second most recent is val, rest are train
ratings_sample['review_date_rank'] = ratings_sample.groupby('reviewerID')['date'].rank(ascending=False)
testset = ratings_sample[ratings_sample['review_date_rank'] == 2].copy()
valset = ratings_sample[ratings_sample['review_date_rank'] == 1].copy()
trainset = ratings_sample[ratings_sample['review_date_rank'] > 2].copy()

reader = Reader(rating_scale = (1.0, 5.0))
train_data = Dataset.load_from_df(trainset[['reviewerID','itemID','rating']], reader)
val_data = Dataset.load_from_df(valset[['reviewerID','itemID','rating']], reader)
test_data = Dataset.load_from_df(testset[['reviewerID','itemID','rating']], reader)
train_sr = train_data.build_full_trainset()
val_sr_before = val_data.build_full_trainset()
val_sr = val_sr_before.build_testset()
test_sr_before = test_data.build_full_trainset()
test_sr = test_sr_before.build_testset()
bsl_options = {'method': 'als', 'n_epochs':3}
bias_baseline = BaselineOnly(bsl_options)
bias_baseline.fit(train_sr)
predictions = bias_baseline.test(val_sr)


RMSE_tune = {}
n_epochs = [10, 15, 25]  # the number of iteration of the SGD procedure
lr_all = [0.002, 0.003, 0.005, 0.008] # the learning rate for all parameters
reg_all =  [0.4, 0.5, 0.6, 0.7, 0.8] # the regularization term for all parameters
# n_epochs = [25]  # the number of iteration of the SGD procedure
# lr_all = [0.005] # the learning rate for all parameters
# reg_all =  [0.02] # the regularization term for all parameters
for n in n_epochs:
    for l in lr_all:
        for r in reg_all:
            print(f'Starting n={n}, l={l}, r={r}')
            algo = SVDpp(n_epochs = n,  &mowitlr_all = l, reg_all = r)
            algo.fit(train_sr)
            predictions = algo.test(val_sr)
            RMSE_tune[n,l,r] = accuracy.rmse(predictions)

min(RMSE_tune.values())

algo_real = SVD(n_epochs = 25, lr_all = 0.008, reg_all = 0.8) # SVD best
# algo_real = SVDpp(n_epochs = 50, lr_all = 0.003, reg_all = 0.8) # SVDpp best
# algo_real = SVDpp(n_epochs = 25, lr_all = 0.005, reg_all = 0.5) # SVDpp best after switching train/val set
algo_real.fit(train_sr)
predictions = algo_real.test(test_sr)
accuracy.rmse(predictions)

actualtestset = pd.read_csv('/Users/joshkennedy/GitKraken/cmu_msba_2022_ml_applications_2/data/pairs_Rating.txt')
actualtestset[['reviewerID','itemID']] = actualtestset['reviewerID-itemID'].str.split("-",expand=True)
actualtestset['rating'] = 0
actual_test_data = Dataset.load_from_df(actualtestset[['reviewerID','itemID','rating']], reader)
actual_test_sr_before = actual_test_data.build_full_trainset()
actual_test_sr = actual_test_sr_before.build_testset()
actual_predictions = algo_real.test(actual_test_sr)
accuracy.rmse(actual_predictions)

actual_pred_dict = {}
for i in actual_predictions:
    actual_pred_dict[i[0]] = i[3]

def user_pred_function(row):
    user = row['reviewerID']
    item = row['itemID']
    try:
        num_user_ratings = num_user_ratings_dict[user]
    except:
        num_user_ratings = 0
    if (num_user_ratings <= 2) and (item in list(df['itemID'].unique())) and (num_item_ratings_dict[item] > 5):
        return avg_item_ratings_dict[item]
    else:
        return actual_pred_dict[user]

actual_test_output = actualtestset.copy()
actual_test_output['prediction'] = actual_test_output.apply(user_pred_function, axis=1)
actual_test_output[['reviewerID-itemID','prediction']].to_csv('jk-rating-output-2021-12-12-v12.csv')