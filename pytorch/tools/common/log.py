import pandas as pd


class ClsLog(object):
    def __init__(self, log_dict):
        self.log_dict = log_dict

        columns = ['epoch', 'time', 'lr', 'train_loss', 'train_acc', 'test_loss']
        for log in log_dict.keys():
            columns.append(log)

        self.columns = columns
        self.log_df = pd.DataFrame(columns=columns)

    def save(self, log_item_dict):
        # log_item_dict = {'train_loss':[2], 'train_acc':[3], ...}
        item_dict = log_item_dict.copy()
        for log in log_item_dict.keys():
            if log not in self.columns:
                item_dict.pop(log)

        item_df = pd.DataFrame.from_dict(item_dict)
        self.log_df = self.log_df.append(item_df, sort=False)

    def write(self, path):
        self.log_df.to_csv(path, sep='\t', index=False, float_format='%.4g')

    def read(self, path):
        read_df = pd.read_csv(path, sep='\t')
        self.log_df.drop(self.log_df.index, inplace=True)
        self.log_df = self.log_df.append(read_df, sort=False)

    def __str__(self):
        return self.log_df.__str__()

