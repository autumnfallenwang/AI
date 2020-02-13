import os
import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split


class ClsLabel(pd.DataFrame):
    def __init__(self, label_path='', data=None):

        def init_label(label_path):
            label_type = osp.splitext(label_path)[1].strip('.')
            if label_type == 'csv':
                label_df = pd.read_csv(label_path, sep=',', header=0, names=['image', 'label'])
            elif label_type == 'txt':
                label_df = pd.read_table(label_path, sep=' ', header=None, names=['image', 'label'])
            else:
                print('Invalid File Type!')
                label_df = None
            return label_df

        if label_path:
            label_df = init_label(label_path)
        else:
            label_df = data
        super().__init__(label_df)
        self.label_path = label_path

    def to_csv(self, path):
        pd.DataFrame.to_csv(self, path, sep=',', index=False)

    def to_txt(self, path):
        pd.DataFrame.to_csv(self, path, sep=' ', index=False, header=None)

    def count(self):
        return self['label'].value_counts().sort_index()

    def shuffle(self, random_seed=233):
        return ClsLabel(data=self.sample(frac=1, random_state=random_seed).reset_index(drop=True))

    def split(self, test_rate=0.2, random_seed=233):
        train_df, test_df = train_test_split(self, test_size=test_rate,
                                             random_state=random_seed,
                                             stratify=self['label'])
        return ClsLabel(data=train_df), ClsLabel(data=test_df)

    def split3(self, split_rate=[0.6, 0.2, 0.2], random_seed=233):
        assert(sum(split_rate) == 1.0)

        if split_rate[2] != 0.0:
            trainval_cl, test_cl = self.split(split_rate[2], random_seed)
            valid_rate = split_rate[1] / (split_rate[0] + split_rate[1])
            train_cl, valid_cl = trainval_cl.split(valid_rate, random_seed)
        else:
            train_cl, valid_cl = self.split(split_rate[1], random_seed)
            test_cl = ClsLabel(data=pd.DataFrame(columns=['image', 'label']))
        return train_cl, valid_cl, test_cl

    def balance(self, method='oversample_times', random_seed=233):
        max_count = max(self.count())

        if method == 'oversample_random':
            balance_df = self.groupby(['label']).apply(lambda x: x.sample(max_count, replace=True,
                                                                          random_state=random_seed))

        elif method == 'oversample_times':
            def oversample_times(df, random_seed):
                times = max_count // len(df)
                left = max_count % len(df)
                left_df = df.sample(left, replace=False, random_state=random_seed)
                over_df = pd.concat([*[df]*times, left_df], ignore_index=True)
                return over_df

            balance_df = self.groupby(['label']).apply(lambda x: oversample_times(x, random_seed))

        # elif method == 'undersample': # TODO

        else:
            print('Invalid Method!')
        return ClsLabel(data=balance_df).shuffle(random_seed)

