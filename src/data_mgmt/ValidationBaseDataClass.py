from torch.utils.data.dataset import Dataset

class ValidationDataClass(Dataset):
    """This is a Dataset object from pytorch

    Args:
        df_input (dataframe): dataframe with which to create dataset
    """
    def __init__(self, df_input, transform=None, target_transform=None):
        # set class variables
        self.df_data = df_input
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