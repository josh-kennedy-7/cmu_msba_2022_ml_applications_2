import os
os.getcwd()
os.chdir('cmu_msba_2022_ml_applications_2/src')
import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split
from data_mgmt.BaseDataClass import BaseDataClass

import sys
sys.path.append("..")

rawdf = BaseDataClass._loadUpDf('/Users/joshkennedy/Documents/CMU/ML for Business Applications 2/','train','.json')
df = rawdf.copy()

data = df[['reviewerID','itemID']].copy()
data['review_count'] = 1
data_dummy = data.copy()

df_matrix = pd.pivot_table(data, values='review_count', index='reviewerID', columns='itemID')
# df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())

# d = df_matrix_norm.reset_index() 
# d.index.names = ['scaled_review_freq'] 
# data_norm = pd.melt(d, id_vars=['reviewerID'], value_name='scaled_review_freq').dropna()
# print(data_norm.shape)
# data_norm.head()



def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data

train_data, test_data = split_data(data)
# train_data_dummy, test_data_dummy = split_data(data_dummy)
# train_data_norm, test_data_norm = split_data(data_norm)

user_id = 'reviewerID'
item_id = 'itemID'
users_to_recommend = list(data[user_id])
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset

def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='pearson')
        
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model

name = 'popularity'
target = 'review_count'
pop_dummy = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


name = 'cosine'
target = 'review_count'
cos_dummy = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


actualtestset_raw = pd.read_csv('/Users/joshkennedy/GitKraken/cmu_msba_2022_ml_applications_2/data/pairs_Purchase.txt')
actualtestset = actualtestset_raw.copy()
actualtestset[['reviewerID','itemID']] = actualtestset['reviewerID-itemID'].str.split("-",expand=True)

name = 'cosine'
target = 'review_count'
test_users_to_recommend = list(actualtestset['reviewerID'])
cos_dummy = model(train_data, name, user_id, item_id, target, test_users_to_recommend, n_rec, n_display)


name = 'pearson'
target = 'review_count'
test_users_to_recommend = list(actualtestset['reviewerID'])
pear_dummy = model(train_data, name, user_id, item_id, target, test_users_to_recommend, n_rec, n_display)



models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy']
eval_dummy = tc.recommender.util.compare_models(test_data, models_w_dummy, model_names=names_w_dummy)


final_model = tc.item_similarity_recommender.create(tc.SFrame(data), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target='review_count', similarity_type='cosine')
n_rec = 50
recom = final_model.recommend(users=test_users_to_recommend, k=n_rec)
recom.print_rows(n_display)

df_rec = recom.to_dataframe()
item_dicts = {}
for i in df_rec['reviewerID'].unique():
    item_dicts[i] = [item for item in df_rec[df_rec['reviewerID'] == i]['itemID']]

df_rec_filtered = df_rec.copy()
# df_rec_filtered = df_rec[df_rec['score'].round(2) >= 0.01].copy()


### attempt #1: highest ranked
combined_test_set = pd.merge(actualtestset,df_rec_filtered, how='left',on=['reviewerID','itemID'])
combined_test_set['rank'] = combined_test_set.groupby('reviewerID-itemID')['score'].rank(method='max',ascending=False)
combined_test_set['filter'] = combined_test_set['rank'].apply(lambda x: x == 1 or pd.isnull(x))
combined_test_set = combined_test_set[combined_test_set['filter']]
combined_test_set['prediction'] = combined_test_set['score'].apply(lambda x: 1 if not pd.isnull(x) else 0)

output = combined_test_set[['reviewerID-itemID','prediction']].copy()
output = pd.merge(actualtestset_raw['reviewerID-itemID'],output, how='left',on=['reviewerID-itemID'])
output['prediction'] = output['prediction'].apply(lambda x: 0 if pd.isnull(x) else x)
output['prediction'].value_counts()
output.to_csv('jk-purchase-output-2021-12-11-v3.csv')



### attempt #2: if the item exists at all in the top 10
output = actualtestset.copy()

def search_item(x):
    if x['itemID'] in item_dicts[x['reviewerID']]:
        return True
    else:
        return False

output['match'] = output.apply(search_item, axis=1)
output['prediction'] = output['match'].apply(lambda x: 1 if x is True else 0)
output['prediction'].value_counts()
output[['reviewerID-itemID','prediction']].to_csv('jk-purchase-output-2021-12-11-v5.csv')


df[df['itemID'] == 'I585050057']

### attempt #3: baseline method
top_ranked_items = df.groupby('itemID').agg({'rating':['count','mean']}).reset_index()
top_ranked_items.columns = ['itemID','num_ratings','mean_rating']
top_ranked_items['num_ratings_pct'] = top_ranked_items['num_ratings'].rank(pct=True)
top_ranked_items[top_ranked_items['num_ratings_pct'] >= 0.5] = top_ranked_items['mean_rating'].rank(pct=True)

min_ratings = 5
top_items_dict = {}
cols = ['mean_rating','num_ratings_pct','mean_ratings_pct','num_ratings']
for c in cols:
    top_items_dict[c] = {}

for i in top_ranked_items['itemID'].unique():
    for col in cols:
        top_items_dict[col][i] = top_ranked_items[top_ranked_items['itemID'] == i][col].values[0]

output = actualtestset.copy()

min_dict = {
    'num_ratings_pct' : 0.8,
    'mean_ratings_pct' : 0.8,
}

def item_filter(x):
    itemID = x['itemID']
    filter_col = 'num_ratings_pct'
    min_ratings = 5
    if itemID in top_items_dict[filter_col].keys():
        if (top_items_dict[filter_col][itemID] >= min_dict[filter_col]) and (top_items_dict['num_ratings'][itemID] >= min_ratings):
            return True
        else:
            return False
    else:
        return False

output['match'] = output.apply(item_filter, axis=1)
output['prediction'] = output['match'].apply(lambda x: 1 if x is True else 0)
output['prediction'].value_counts()
output[['reviewerID-itemID','prediction']].to_csv('jk-purchase-output-2021-12-11-v9.csv')