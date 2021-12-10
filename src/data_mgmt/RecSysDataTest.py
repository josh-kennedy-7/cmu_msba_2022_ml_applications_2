from . import ValidationBaseDataClass
import pandas as pd

class TestRecSysData(ValidationBaseDataClass.ValidationDataClass):
    """This is a Dataset object from pytorch

    Args:
        df_input (dataframe): dataframe with which to create dataset
    """
    def __init__(self, df_input, target_file=None,
                    transform=None, target_transform=None):

        # set class variables
        self.transform = transform
        self.target_transform = target_transform

        test_text = pd.read_csv(target_file,delimiter='-')
        self.df_data = TestRecSysData._beatTheCrapOutOfThisData(df_input, test_text)


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

        return features


    @staticmethod
    def _beatTheCrapOutOfThisData(dfin, dfout):
        """
        will return a dataframe that has correlated the reviewer and item
        ID's within the test data set versus ones we recognize from the
        training data set.
        """
        dfout.columns = ['reviewerID','itemID']

        uid_pairs = dfin.set_index('reviewerID').loc[
                        dfout.query('reviewerID in @dfin.reviewerID').reviewerID].loc[
                            dfout.query('reviewerID in @dfin.reviewerID').reviewerID].\
                                uid.drop_duplicates().copy()

        pid_pairs = dfin.set_index('itemID').loc[
                        dfout.query('itemID in @dfin.itemID').itemID].loc[
                            dfout.query('itemID in @dfin.itemID').itemID].\
                                pid.drop_duplicates().copy()

        dfout=dfout.join(uid_pairs,how='left',on='reviewerID')
        dfout=dfout.join(pid_pairs,how='left',on='itemID')
        dfout.uid=dfout.uid.fillna(method='backfill') # kind of a hack for now
        dfout.pid=dfout.pid.fillna(method='backfill')

        return dfout
