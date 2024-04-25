import pandas as pd
from cnrs_utils.linker_FR import linker

# loading
directory_time_series = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP'
file_nlp = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/French_Corpus.csv'
path_knowledge = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/crisis_knowledge_FR.csv'
Knowledge = pd.read_csv(path_knowledge, sep=',')
Nlp_data = pd.read_csv(file_nlp, delimiter='\t', encoding='utf-8')

# choose train / test
# All_crisis = ['Wildfire ', 'Earthquake', 'Haze', 'Flood', 'Other', 'Terrorist_attack', 'Huricane']
All_crisis = ['Storms', 'Flood', 'Fire', 'Hurricane', 'Explosion', 'ATTACK', 'Collapse']
Test_crisis = ['Storms']

nlp_crisis = Nlp_data['type_crisis']
type_counts = nlp_crisis.value_counts()

nlp_train, nlp_test = Nlp_data[~Nlp_data.type_crisis.isin(Test_crisis)], Nlp_data[Nlp_data.type_crisis.isin(Test_crisis)]

# Train_da = Knowledge[Knowledge['Crisis_Type'] != Test_crisis]
# Test_knowledge = Knowledge[Knowledge['Crisis_Type'] == Test_crisis]

# link the data
train_multi_modal = linker(nlp_train,
                           Knowledge,
                           directory_time_series,
                           5,
                           'Crisis_Predictability',
                           'label',
                           'date',
                           0)

test_multi_modal = linker(nlp_test,
                          Knowledge,
                          directory_time_series,
                          5,
                          'Crisis_Predictability',
                          'label',
                          'date',
                          0)

print(train_multi_modal.head())
print(test_multi_modal.head())


# def flatten_extend(matrix):
#     flat_list = []
#     for row in matrix:
#         flat_list.extend(row)
#     return flat_list
#
# train_features = list(train_multi_modal['Window'])
# test_features = list(test_multi_modal['Window'])
#
# for i in range(len(train_features)):
#     train_features[i] = flatten_extend(train_features[i])
#     train_features[i] = [float(j) for j in train_features[i]]

end = 1