import pandas as pd
import os
import json


# this function put on the good format the date from the nlp dataset
def to_date(my_date):
	set_month= {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
				'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
	month = set_month[my_date[4:7]]
	day = my_date[8:10]
	year = my_date[26:30]
	final_date = year+'-'+month+'-'+day
	return final_date


# open the knowledge base of the corresponding state with the corresponding category
def open_state_file(dataset_path: str, STATE: str, CATEGORY: str) -> list[str]:
	tags = set()
	with open(dataset_path+'/'+STATE+'/'+STATE+'_keywords_'+CATEGORY+'_no_dupe.txt', 'r') as fp:
		for line in fp:
			# remove linebreak from a current name
			# linebreak is the last character of each line
			x = line[:-1]
			# add current item to the list
			tags.add(x.lower())
	return tags


# the function take a time series dataframe (df) and return an other dataset with all the data put in one parameter
# (Window) except the date and the label which are in different columns
# window size is equal to the size of the window
def window_creation(df: object, window_size: int, label_column: str, timestamp_colum: str) -> object:
	# which row we want to put in our windows (all minus label and timestamp)
	good_row = list(df.columns)
	good_row.remove(label_column)
	good_row.remove(timestamp_colum)
	window_df = df[good_row]
	label_df = df[label_column]
	date_df = df[timestamp_colum]
	window_data = []
	label_data = []
	date_data = []
	for i in range(len(df)-window_size):
		my_window = window_df.loc[i:i+window_size-1]
		window_data.append(my_window.to_numpy())
		label_data.append((label_df.loc[i+window_size-1]))
		date_data.append((date_df.loc[i+window_size-1]))
	return_value = pd.DataFrame({'Date': date_data, 'Window': window_data, 'label': label_data})
	return return_value


# the function that link the NLP data and the time series data for English dataset
# crisis_to_link is one row from crisis knowledge
# directory is the path of NLP data
# dataset_path  is the path of time series data
# window_size is the size of the window for time series data
# fillNA is the value to fill the missing value
def linker(
		crisis_to_link: object,
		directory: str,
		dataset_path: str,
		window_size: int,
		label_column: str,
		date_column: str,
		fillNA : int) -> object:
	# make a list of which state contain the crisis
	states = crisis_to_link['Places'].replace('[', '').replace(']', '').split(',')
	# which features we want to keep for our experiments
	features = ['date','mean_speed_wind','precipitation','snow_fall','snow_depth','temp_max','temp_min','temp_mean','wind_dir_angle_2min','wind_dir_angle_5sec','wind_max_speed_2min','wind_max_speed_5sec','label_Sudden']
	# the return object
	df_return = pd.DataFrame({'Date': [], 'Text': [], 'Batch': [], 'label': []})
	for state in states:
		# we open the time series file corresponding to the state
		my_features = pd.read_csv(dataset_path+'/'+state+'/'+state+'_mean_data_LLM.csv')
		# filter to keep the features
		my_features = my_features[features]
		my_features = my_features.fillna(fillNA)
		# we create the window for the time series data
		window_df = window_creation(my_features,window_size,label_column,date_column) 
		# we get the keywords to link the location mention to the state
		tags_city = open_state_file(dataset_path,state,'city')
		tags_county = open_state_file(dataset_path,state,'county')
		tags_state = open_state_file(dataset_path,state,'state')
		date = []
		text = []
		label_final = []
		window_data = []
		# now the Time series data and the knowledge base required to link the data are ready
		# we process the NLP data and link to the time series data
		for root, dirs, files in os.walk(directory+'/'+crisis_to_link['Path_name']):
			for fil in files:
				if fil.endswith('.jsonl'):
					# we load jsonl files
					with open(os.path.join(root,fil), 'r') as json_file:
						json_list = list(json_file)
						# for each tweet loaded
						for json_str in json_list:
							result = json.loads(json_str)
							# the location mention
							place = result['location_mentions']
					# if the location mention is empty (the tweet does not refer to particular place),
					# thanks to the crisis_to_link, we know which crisis this tweet make reference
					# (the tweet speak about this crisis) so we assume that if location mention is empty
					# we assume that the tweet make a reference to the current state since this state is the localisation of the crisis
							if place == []:
								# we put NLP date on the same format as time series date
								date_NLP = to_date(result['created_at'])
								# we check if there is matching date between time series and NLP
								if list(window_df['Window'][ window_df['Date'] == date_NLP]) != [] :
									# the date
									date.append(date_NLP)
									# the nlp data
									text.append(result['text'])
									# time series data
									linked_data = window_df[ window_df['Date'] == date[-1]]
									my_window = list(linked_data['Window'])[0]
									window_data.append(my_window)
									# for the label, we take reference from the time series label
									label_final.append(linked_data['label'])

								for zone in place:
									# if the location mention refer to a state and this reference is in our knowledge database
									bool1 = zone['type'] == 'State' and zone['text'].lower() in tags_state
									# if the location mention refer to a county and this reference is in our knowledge database
									bool2 = zone['type'] == 'County' and zone['text'].lower() in tags_county
									# if the location mention refer to a city and this reference is in our knowledge database
									bool3 = zone['type'] == 'City/town' and zone['text'].lower() in tags_city
									if bool1 or bool2 or bool3:
										date.append(to_date(result['created_at']))
										text.append(result['text'])
										#time series data
										linked_data = window_df[window_df['Date'] == date[-1]]
										my_window = list(linked_data['Window'])[0]
										window_data.append(my_window)
										#for the label, we take reference from the time series label
										label_final.append(linked_data['label'])
		df = pd.DataFrame({'Date': date, 'Text': text, 'Window': window_data, 'label': label_final})
		df_return = pd.concat([df_return, df])
	return df_return
