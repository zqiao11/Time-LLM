from torch.utils.data import Dataset
import pandas as pd

class MMDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        date = row['Date']
        text = row['Text']
        window = row['Window']
        label = row['label']
        return (date, text, window, label)



def out_of_type_train_test(knowledge, test_crisis, linker, directory_nlp, directory_time_series):
    train_knowledge = knowledge[knowledge['Crisis'] != test_crisis]
    test_knowledge = knowledge[knowledge['Crisis'] == test_crisis]

    train_df = pd.DataFrame({'Date': [], 'Text': [], 'label': []})
    for index, crisis in train_knowledge.iterrows():
        train_df = pd.concat([train_df,
                              linker(crisis,
                                     directory_nlp,
                                     directory_time_series,
                                     12,
                                     'label_Sudden',
                                     'date',
                                     0)]
                             )

    test_df = pd.DataFrame({'Date': [], 'Text': [], 'label': []})
    for index, crisis in test_knowledge.iterrows():
        test_df = pd.concat([test_df,
                             linker(crisis,
                                    directory_nlp,
                                    directory_time_series,
                                    12,
                                    'label_Sudden',
                                    'date',
                                    0)]
                            )

    return train_df, test_df
