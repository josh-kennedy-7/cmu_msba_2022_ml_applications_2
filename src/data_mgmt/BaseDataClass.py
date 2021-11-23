from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import pandas as pd
import torch
import gzip


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
        # category_results = BaseDataClass._assessDirectory(root_dir)
        category_results = "json_gz_only"

        if category_results == "json_gz_only":
            self.df_data_in = BaseDataClass._loadUpDf(
                root_dir, data_file_name, ".json.gz")
        elif category_results == "json_only":
            self.df_data_in = BaseDataClass._loadUpDf(
                root_dir, data_file_name, ".json")
        elif category_results == "saved_files":
            pass
            # TODO -- REED -- replace this with a thing that just reads the saved dataframe
        else:
            print("ERROR: BaseDataClass did not know where to find category results!")

        self.df_data_in.to_csv(root_dir+data_file_name+".csv")

        self.transform = transform
        self.target_transform = target_transform

    # TODO -- ARA -- rewite for our class

    def __len__(self):
        """The __len__ function returns the number of samples in our dataset.

        Simple enough right?
        """
        return len(self.img_labels)

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
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

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

        return df


# TODO -- REED -- redo this code for testing purposes
if __name__ == "__main__":
    ppath = "C:\\git\\cmu_msba_2022_ml_applications_2\\data\\"
    transformed_dataset = BaseDataClass(ppath)

    # dataloader = DataLoader(transformed_dataset, batch_size=4,
    #                         shuffle=True, num_workers=0)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     if i_batch == 3:
    #         print(sample_batched[0])
    #         print(sample_batched[1])
    #         break
