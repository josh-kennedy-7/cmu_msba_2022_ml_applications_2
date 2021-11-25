from torch.utils.data.dataset import Dataset
import pandas as pd
import gzip
import os


class BaseDataClass(Dataset):
    """This is a Dataset object from pytorch for our covid dataset

    Args:
        csv_file (location): path to the data
    """

    def __init__(self, root_dir, train_or_test="train", transform=None, target_transform=None):
        """The __init__ function is run once when instantiating the Dataset object.

        We initialize the directory containing the images, 
        the annotations file, and both transforms.
        """

        # TODO -- REED -- super ideal behavior would be to
        # A) look in the dir, if there's
        #   1) saved dataframes or csv load those
        #   2) jsons then load those, and save dataframes
        #   3) json.gz then unpack and move up to 2
        if train_or_test == "train":
            data_file_name = "train"
        else:
            data_file_name = "test_Category"

        # TODO -- REED -- write this function
        category_results = BaseDataClass._assessDirectory(
            root_dir, data_file_name)
        # category_results = "json_gz_only"

        save_results = True

        if category_results == "json_gz_only":
            self.df_data_in = BaseDataClass._loadUpDf(
                root_dir, data_file_name, ".json.gz")
        elif category_results == "json_only":
            self.df_data_in = BaseDataClass._loadUpDf(
                root_dir, data_file_name, ".json")
        elif category_results == "saved_files":
            self.df_data_in = pd.read_csv(root_dir+data_file_name+".csv")
            save_results = False
        else:
            print("ERROR: BaseDataClass did not know where to find category results!")

        if save_results:
            self.df_data_in.to_csv(root_dir+data_file_name+".csv")

        self.transform = transform
        self.target_transform = target_transform

    # TODO -- ARA -- rewite for our class

    def __len__(self):
        """The __len__ function returns the number of samples in our dataset.

        Simple enough right?
        """
        return len(self.df_data_in)

    # TODO -- ARA -- rewite for our class

    def __getitem__(self, idx):
        """The __getitem__ function loads and returns a sample from the dataset at the given index idx.

        Based on the index, it identifies the image’s location on disk, 
        converts that to a tensor using read_image, retrieves the 
        corresponding label from the csv data in self.img_labels, calls 
        the transform functions on them (if applicable), and returns 
        the tensor image and corresponding label in a tuple.

        OUTPUT WILL BE IN FEATURES, LABEL FORMAT

        Args:
            idx -- the index you want the thingy of
        """
        return self.df_data_in.iloc[idx]

    @staticmethod
    def _assessDirectory(root_dir, f_name):
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
        try:  # if using colab
            # Mounting google drive
            from google.colab import drive
            drive.mount("/content/gdrive")

            local_path = pathname+filename+extensions
            option = 'gdrive'
        except:
            local_path = pathname+filename+extensions
            option = 'local'

        compiledRatings = dict()

        if extensions == '.json.gz':
            func_choice = BaseDataClass._readGz(local_path)
        elif extensions == '.json':
            func_choice = BaseDataClass._readJson(local_path)

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
        
        df.df_data_in.\
            reset_index(inplace=True).\
            rename(columns={'index':'reviewHash'})

        return df


def main(ppath="C:\\git\\cmu_msba_2022_ml_applications_2\\data\\"):
    transformed_dataset = BaseDataClass(ppath, train_or_test="train")

    # dataloader = DataLoader(transformed_dataset, batch_size=4,
    #                         shuffle=True, num_workers=0)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     if i_batch == 3:
    #         print(sample_batched[0])
    #         print(sample_batched[1])
    #         break

# TODO -- REED -- redo this code for testing purposes
if __name__ == "__main__":
    main()


