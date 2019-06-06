import pandas as pd
import os
import tarfile
import logging

logging.basicConfig(level=logging.INFO)

class ArxivData():
    def __init__(self, path):
        assert os.path.exists(path)
        self.path = path
        self.num_classes = 10
        self._get_data()

    def _read_tar(self):
        """
            Returns a pandas dataframe of the provided arxiv data file.
        """
        tar = tarfile.open(self.path, "r:gz")
        fd = tar.extractfile(tar.getmembers()[-1])
        df = pd.read_csv(fd, sep='\t')
        tar.close()
        return df

    def _process_dataframe(self, df):
        """
            Drops data that isn't part of the top k classes and adds abstract
            and column data together into a column.
        """
        # Count tag frequency
        tag_counts = {}
        tags = df['primary_cat']
        for idx, tag in enumerate(tags):
            if tag not in tag_counts:
                tag_counts[tag] = 0
            tag_counts[tag] += 1

        count_series = pd.Series(tag_counts)
        count_series = count_series.sort_values()
        allowed_tags = count_series[-1 * self.num_classes:]

        # Use only entries that are allowed
        allowed_df = df[df.primary_cat.isin(allowed_tags.keys())]
        allowed_df = allowed_df.reset_index()

        # Add title to abstract and process full list of categories
        allowed_df['train_text'] = allowed_df['title'] + '. ' + allowed_df['abstract']
        allowed_df['category_list'] = allowed_df['categories'].apply(lambda row: row.split(','))

        return allowed_df

    def _get_data(self):
        """
            Read in, preprocess, and split data into a 70/20/10 train/dev/test split.
        """
        logging.info(" Reading data from {}...".format(self.path))
        df = self._read_tar()

        logging.info(" Processing and splitting data...")
        df = self._process_dataframe(df)

        # Get dev set
        dev_df = df.sample(frac=0.2, random_state=0)
        train_df = df.drop(dev_df.index)

        # Get train/test sets
        test_df = train_df.sample(frac=(0.1/0.8), random_state=0)
        train_df = train_df.drop(test_df.index)

        # Reindex each df.
        train_df = train_df.reset_index()
        dev_df = dev_df.reset_index()
        test_df = test_df.reset_index()

        assert df.shape[0] == dev_df.shape[0] + train_df.shape[0] + test_df.shape[0]

        # Train data
        train_data = list(train_df.train_text.values)
        train_labels = list(train_df.primary_cat.values)
        train_all_labels = list(train_df.category_list.values)

        # Dev data
        dev_data = list(dev_df.train_text.values)
        dev_labels = list(dev_df.primary_cat.values)
        dev_all_labels = list(dev_df.category_list.values)

        # Test data
        test_data = list(test_df.train_text.values)
        test_labels = list(test_df.primary_cat.values)
        test_all_labels = list(test_df.category_list.values)


        self.train_data, self.train_labels, self.train_all_labels = train_data, train_labels, train_all_labels
        self.dev_data, self.dev_labels, self.dev_all_labels = dev_data, dev_labels, dev_all_labels 
        self.test_data, self.test_labels, self.test_all_labels = test_data, test_labels, test_all_labels

if __name__ == '__main__':
    data = ArxivData("data/2018.tar.gz")
