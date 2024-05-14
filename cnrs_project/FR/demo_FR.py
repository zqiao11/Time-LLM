import pandas as pd
import numpy as np
from cnrs_project.FR.linker_FR import linker

# loading
#directory_time_series = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP'
file_nlp = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/French_Corpus.csv'
#path_knowledge = '/home/eee/qzz/datasets/CNRS/Crisis-TS-NLP/crisis_knowledge_FR.csv'

#CHANGE : with one csv multimodal csv file, you only need to load one file
#loading
file_data = '../../moredata/romainM/multi_modal_french.csv'

multi_modal_data = pd.read_csv(file_data,sep="\t")

#the window is saved as string we need to rechange the type to np array
list_of_time_series_data = []

for row in multi_modal_data["Window"]:
	a = row.replace("[","")
	a = a.replace("]","")
	a = a.split("\n")
	for i in range(len(a)) :
		while '  ' in a[i] :
			a[i]= a[i].replace('  ',' ')
		a[i]= a[i].strip(' ')
		a[i]= a[i].split(" ")
		for j in range(len(a[i])):
			a[i][j]= float(a[i][j])
	list_of_time_series_data.append(np.array(a))

multi_modal_data['Window']= list_of_time_series_data
#Knowledge = pd.read_csv(path_knowledge, sep=',')  # (13, 5)
Nlp_data = pd.read_csv(file_nlp, delimiter='\t', encoding='utf-8')  # (19595, 62)

# choose train / test
# All_crisis = ['Wildfire ', 'Earthquake', 'Haze', 'Flood', 'Other', 'Terrorist_attack', 'Huricane']
All_crisis = ['Storms', 'Flood', 'Fire', 'Hurricane', 'Explosion', 'ATTACK', 'Collapse']
Test_crisis = ['Storms']

nlp_crisis = Nlp_data['type_crisis']
type_counts = nlp_crisis.value_counts()

train_multi_modal, test_multi_modal = multi_modal_data[~multi_modal_data.Crisis_Type.isin(Test_crisis)], multi_modal_data[multi_modal_data.Crisis_Type.isin(Test_crisis)]


print(train_multi_modal.head())
print(test_multi_modal.head())

end = 1