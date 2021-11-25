import os
import pandas as pd
import numpy as np
import gzip

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

def readJSON(f):
    for l in open(f):
        yield eval(l)

def load_data_to_df(local_path="/Users/joshkennedy/Documents/CMU/ML for Business Applications 2/train.json"):
    try: # if using colab
        #Mounting google drive
        from google.colab import drive
        drive.mount("/content/gdrive")

        local_path = "/content/gdrive/MyDrive/MachineLearning_Fall21/Raw_Data/train.json.gz"
        option = 'gdrive'
    except:
        local_path = local_path
        option = 'local'

    compiledRatings = dict()

    func_choice = readJSON(local_path) if option == 'local' else readGz(local_path)

    # ----------------------- #
    # Transform the json to a dataframe
    # ----------------------- #
    for l in func_choice:
        row = l['reviewHash']
        userRating = dict()

        userRating['reviewTime'] = l['reviewTime']
        userRating['reviewText'] = l['reviewText']
        userRating['nHelpful'] = l['helpful']['nHelpful']
        userRating['outOf'] = l['helpful']['outOf']
        userRating['reviewerID'] = l['reviewerID']
        userRating['reviewHash'] = l['reviewHash']
        userRating['categories'] = l['categories']
        userRating['unixReviewTime'] = l['unixReviewTime']
        userRating['itemID'] = l['itemID']
        userRating['rating'] = l['rating']
        userRating['summary'] = l['summary']
        userRating['categoryID'] = l['categoryID']
        
        compiledRatings[row] = userRating

    df_train_raw = pd.DataFrame(compiledRatings).T
    df_train = df_train_raw.copy()

    # ----------------------- #
    # Convert columns
    # ----------------------- #
    dtype_dict = {
        'nHelpful': 'int64',
        'outOf': 'int64',
        'rating': 'float',
        'reviewTime': 'datetime64'
    }

    df_train = df_train.astype(dtype_dict)
    
    # ----------------------- #
    # create new columns out of existing data
    # ----------------------- #
    df_train['helpfulPerc'] = df_train['nHelpful'] / df_train['outOf']
    df_train['summaryCharacterLength'] = df_train['summary'].apply(lambda x: len(x))
    df_train['reviewTextCharacterLength'] = df_train['reviewText'].apply(lambda x: len(x))

    # ----------------------- #
    # If we wanted to create category specific columns:
    # ----------------------- #
    df_train['parentCategory'] = df_train['categories'].apply(lambda x: x[0][0])
    df_train['cat1'] = df_train['categories'].apply(lambda x: x[0][1])
    df_train['cat2'] = df_train['categories'].apply(lambda x: x[0][2] if len(x[0]) > 2 else '')
    df_train['cat3'] = df_train['categories'].apply(lambda x: x[0][3] if len(x[0]) > 3 else '')
    df_train['cat1_child'] = df_train['categories'].apply(lambda x: x[0][-1])
    df_train['cat2_parent'] = df_train['categories'].apply(lambda x: x[1][0] if len(x) > 1 else '')
    df_train['cat2_child'] = df_train['categories'].apply(lambda x: x[1][-1] if len(x) > 1 else '')

    return df_train
