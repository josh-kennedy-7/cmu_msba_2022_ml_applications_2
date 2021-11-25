from torch.utils.data.dataset import Dataset
import pandas as pd
import gzip
import os
from tqdm import tqdm


class BaseDataClass(Dataset):
    """This is a Dataset object from pytorch for our covid dataset

    Args:
        csv_file (location): path to the data
    """

    def __init__(self, root_dir, data_file_name="train", transform=None, target_transform=None):
        """The __init__ function is run once when instantiating the Dataset object.

        We initialize the directory containing the images, 
        the annotations file, and both transforms.
        """

        if data_file_name != "train":
            raise Exception("This object only works on training data right now, blame Reed.")

        category_results = BaseDataClass._assessDirectory(
            root_dir, data_file_name)
        save_results = True

        if category_results == "json_gz_only":
            df_read_out = BaseDataClass._loadUpDf(
                root_dir, data_file_name, ".json.gz")
        elif category_results == "json_only":
            df_read_out = BaseDataClass._loadUpDf(
                root_dir, data_file_name, ".json")
        elif category_results == "saved_files":
            df_read_out = pd.read_csv(root_dir+data_file_name+".csv")
            save_results = False
        else:
            raise Exception("BaseDataClass did not know where to find category results!")

        if save_results:
            df_read_out.to_csv(root_dir+data_file_name+".csv")
            
        # set class variables
        self.df_data = df_read_out
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """The __len__ function returns the number of samples in our dataset.

        Simple enough right?
        """
        return len(self.df_data)

    def __getitem__(self, idx):
        """The __getitem__ function loads and returns a sample from the dataset at the given index idx.

        Note that if you're reading this you are in the Base Class of our
        dataset class which means the return of __getitem__() will not be
        a (feature,label) pair, it's just going to be the row.
        
        If you want to feed this into a dataloader to start working with it
        then you're going to need to have:
            -> A transform() function
            -> A transform_target() function
            
        ...which hopefully by the time you're reading this Reed will have
        authored for most of the challenges...
        
        And instead of using this base class you're going to want to use
        the overloaded class specific to the challenge you're trying
        to address!!!

        Args:
            idx -- the index you want the thingy of
            
        Returns:
            tuple -- index 0 is features, index 1 is targets/labels/whateva
        """
        reqested_row = self.df_data.iloc[idx].copy()
        
        if self.transform:
            features = self.transform(reqested_row)
        else:
            features = reqested_row
            
        if self.target_transform:
            label = self.target_transform(reqested_row)      
        else:
            label = None
        
        return (features, label)
    

    @staticmethod
    def _assessDirectory(root_dir, f_name):
        """looks at the data directory to determine the type and format of data accessible by the dataset object. This will save time because if you've run it once you'll have a .csv and you won't need to reformat the data from the .json or .json.gz format!!

        Args:
            root_dir (str): file path to the data directory
            f_name (str): name of the file to search for

        Returns:
            str: string describing the available data to access
        """
        result = None
        
        found_json      =   False
        found_json_gz   =   False
        with os.scandir(root_dir) as diriter:
            for this_file in diriter:
                if f_name + ".csv" == this_file.name:
                    result = "saved_files"
                    break
                elif f_name + ".json" == this_file.name:
                    found_json = True
                    continue                
                elif f_name + ".json.gz" == this_file.name:
                    found_json_gz = True
                    continue
                
        if found_json:
            result = "json_only"
        elif found_json_gz:
            result = "json_gz_only"
            
        if not result:
            raise Exception("No recognizable file type found within data directory.")
            
        return result
            

    @staticmethod
    def _readGz(f):
        for l in gzip.open(f):
            yield eval(l)

    @staticmethod
    def _readJson(f):
        for l in open(f):
            yield eval(l)

    @staticmethod
    def _loadUpDf(pathname, filename, extensions):
        """Governs the transfer logic from stored data (.json.gz, .json, or .csv) over to the dataframe for use within the dataset object.

        At some point this will have contingency planning to load up the test data sets
        but for now it only works on the train.json.gz suite of sets.

        Args:
            pathname (str): path where the data files reside
            filename (str): name of the file to load from
            extensions (str): must be ".json.gz", ".json", or ".csv" (auto set)

        Returns:
            Pandas Dataframe: dataframe with the data to be machine-learned
        """
        local_path = pathname+filename+extensions
        compiledRatings = dict()

        if extensions == '.json.gz':
            func_choice = BaseDataClass._readGz(local_path)
        elif extensions == '.json':
            func_choice = BaseDataClass._readJson(local_path)

        # ----------------------- #
        # Transform the json to a dataframe
        # ----------------------- #
        for l in tqdm(func_choice,desc="Loading JSON into Dataframe"):
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

        df_raw = pd.DataFrame(compiledRatings).T
        df = df_raw.copy()

        # ----------------------- #
        # Convert columns
        # ----------------------- #
        dtype_dict = {
            'nHelpful': 'int64',
            'outOf': 'int64',
            'rating': 'float',
            'reviewTime': 'datetime64'
        }

        df = df.astype(dtype_dict)

        # ----------------------- #
        # create new columns out of existing data
        # ----------------------- #
        df['helpfulPerc'] = df['nHelpful'] / df['outOf']
        df['summaryCharacterLength'] = df['summary'].apply(
            lambda x: len(x))
        df['reviewTextCharacterLength'] = df['reviewText'].apply(
            lambda x: len(x))

        # ----------------------- #
        # If we wanted to create category specific columns:
        # ----------------------- #
        df['parentCategory'] = df['categories'].apply(
            lambda x: x[0][0])
        
        df = df.reset_index()
        df = df.rename(columns={'index':'reviewHash'})

        return df


def main(ppath="C:\\git\\cmu_msba_2022_ml_applications_2\\data\\"):
    transformed_dataset = BaseDataClass(ppath)
    print(transformed_dataset.df_train.head())

# TODO -- REED -- redo this code for testing purposes
if __name__ == "__main__":
    main()

