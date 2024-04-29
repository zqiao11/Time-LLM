import pandas as pd
from linker_ENG import linker

# loading file
directory_nlp = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/English_Corpus'
directory_time_series = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/MeteoData-US'
path_knowledge = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/crisis_knowledge_EN.csv'
knowledge = pd.read_csv(path_knowledge, sep='\t')  # (9, 3), 3 columns represent Crisis, Places, Path_name (tweets)

# for all test crisis in english
All_possible_crisis = ['Hurricane', 'WildFire', 'Flood']
Test_crisis = 'Hurricane'

Train_knowledge = knowledge[knowledge['Crisis'] != Test_crisis]
Test_knowledge = knowledge[knowledge['Crisis'] == Test_crisis]

# link
multi_modal_test_data = pd.DataFrame({'Date': [], 'Text': [], 'label': []})
for index, crisis in Test_knowledge.iterrows():
	multi_modal_test_data = pd.concat([multi_modal_test_data,
									   linker(crisis, directory_nlp,
											  directory_time_series,
											  12,
											  'label_Sudden',
											  'date',
											  0)])
	
multi_modal_train_data = pd.DataFrame({'Date': [],'Text': [], 'label': []})
for index, crisis in Train_knowledge.iterrows():
	multi_modal_train_data = pd.concat([multi_modal_train_data,
										linker(crisis,
											   directory_nlp,
											   directory_time_series,
											   12,
											   'label_Sudden',
											   'date',
											   0)]
									   )

print(multi_modal_train_data.head())
print(multi_modal_test_data.head())