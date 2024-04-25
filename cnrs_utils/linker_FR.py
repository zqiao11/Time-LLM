import numpy as np
import pandas as pd
import os
from sklearn import preprocessing


# this function put on the good format the date from the time series dataset
def to_date(my_date):
	my_date = str(my_date)
	set_month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
				 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
	month = my_date[4:6]
	day = my_date[6:8]
	year = my_date[0:4]
	final_date = year+'-'+month+'-'+day
	return final_date


def window_zeros_creation(window_ref):
	return np.zeros(window_ref.shape)


# the function takes a time series dataframe (df) and return another dataset with all the data put in one parameter
# (Window) except the date and the label which are in different columns
# window size is equal to the size of the window
def window_creation(df,size_window,label_column,label_2_column,timestamp_colum):
	good_row = list(df.columns)
	good_row.remove(label_column)
	good_row.remove(label_2_column)
	good_row.remove(timestamp_colum)
	window_df = df[good_row]
	window_df = window_df.reset_index()
	window_df = window_df.drop(columns=['index'])
	label_df = df[label_column]
	label_df = label_df.reset_index()
	label_df = label_df.drop(columns=['index'])
	date_df = df[timestamp_colum]
	date_df = date_df.reset_index()
	date_df = date_df.drop(columns=['index'])
	window_data = []
	label_data = []
	date_data = []
	for i in range(len(df)-size_window):  # Distance is 1.
		my_window = window_df.loc[i:i+size_window-1]  # loc includes both staring and ending index
		window_data.append(preprocessing.normalize(my_window.to_numpy()))
		label_data.append(int(label_df.loc[i+size_window-1]))
		date_data.append(list(date_df.loc[i+size_window-1])[0])

	return_value = pd.DataFrame({'Date': date_data, 'Window': window_data, 'label': label_data})
	return return_value


# the function that link the NLP data and the time series data for French dataset
# text_data is the french NLP dataset on dataframe format
# crisis_knowledge is a dataframe from the crisis_knowledge CSV
# directory is the path the repertory MeteoData
# window size is equal to the size of the window
# fillNA is the value to fill the missing value
def linker(
		text_data: object,
		crisis_knowledge: object,
		directory: str,
		window_size: int,
		label_column_1: str,
		label_column_2: str,
		date_column: str,
		fillNA: int) -> object:
	# select
	time_data = directory+'/MeteoData-FR'  # MeteoDataProcessed-FR
	# we join crisis knowledge and NLP data on the name of the crisis
	text_data = text_data.join(crisis_knowledge.set_index('Crisis Name'), on='event', validate='m:1')
	# features to keep for Time series
	features = ['numer_sta', 'date', 'pmer', 'tend', 'ff', 't', 'u', 'n', 'label', 'Crisis_Predictability']
	list_of_date = []
	list_of_text = []
	list_of_window = []
	list_of_label = []

	# Loop over time series data
	for root, dirs, files in os.walk(time_data):
		for fil in files:
			# fil: long format TS, including ts from different stations and dates in the given month.
			if fil.endswith('.csv'):
				my_features = pd.read_csv(os.path.join(root, fil), sep=';')
				# replace missing values and replace label by number
				my_features = my_features[features].fillna(fillNA).replace({'mq': fillNA, 'No_Crisis': 0, 'Crisis': 1, 'Ecological': 1, 'Sudden': 2})
				# Get all the station id occurred in the period
				set_of_station = set(list(my_features['numer_sta']))
				my_features['date'] = list(map(to_date, list(my_features['date'])))
				set_of_date = set(list(my_features['date']))  # Get the dates included in the period
				dict_by_station = {}
				for station in set_of_station:
					# creation of the windows. Intervals between time steps is 3 hour.
					temp = my_features[my_features['numer_sta'] == station]  # the series of one station
					temp = temp.drop(columns=['numer_sta'])
					window_df = window_creation(temp,window_size, label_column_1, label_column_2, date_column)
					# each time dataframe of window is linked to the related station
					# window_df is in shape of (num_windows, 3), 3 columns are 'Date' 'Window' 'label'.
					# Window is the nparray of a window. label is its int class label.
					dict_by_station[station] = window_df

				# for each tweet
				for index, text_line in text_data.iterrows():
					# if the NLP date can be found in the time series date
					if text_line['date'] in set_of_date:
						# if related station empty (in the case where a tweet is related to a crisis with no knowledge)
						if text_line['Related station'] != text_line['Related station']:  # False if NaN
							list_of_date.append(text_line['date'])
							list_of_text.append(text_line['text'])
							# we create a mask window full of zeros with the same size as other windows
							list_of_window.append(window_zeros_creation(list(window_df['Window'])[-1]))
							# if the message is not usefull then this is not related to a crisis
							if text_line['CAT2'] == 'Message-NonUtilisable' :
								list_of_label.append('Not_Crisis_period')
							else:
								if text_line['type_crisis'] == 'Flood' or text_line['type_crisis'] == 'Hurricane' or text_line['type_crisis'] == 'Storm':
									list_of_label.append('Ecological_crisis')
								else:
									list_of_label.append('Sudden_Crisis')
						else:
							list_of_station = eval(text_line['Related station'])
							for i in range(len(list_of_station)):
								list_of_station[i] = int(list_of_station[i])
							# station related to the crisis
							for station in list_of_station:
								temp_list = []
								# if the station is in the time series data
								if station in dict_by_station.keys() :
									# we take all the data related to this station
									sample_station = dict_by_station[station]
									# we choose only the one with the good date
									good_date = sample_station[sample_station['Date'] == text_line['date']]
									if len(list(good_date['Window'])) != 0:
										for window in good_date['Window']:  # window: (window_size, num_features)
											temp_list.append(window)

							# if there is multiple data with the same day we make the mean of these data
							# Todo: It's not a good practice. Can have multi windows from multi stations.
							if temp_list:
								# Stack the arrays along a new axis
								temp_list_clean = [x for x in temp_list if x is not None and (not isinstance(x, list) or None not in x)]
								
								stacked_array = np.stack(temp_list_clean)
								# Compute the mean along the newly created axis (axis 0)
								mean_array = np.mean(stacked_array, axis=0)
							else:
								mean_array = window_zeros_creation(list(window_df['Window'])[0])

							list_of_date.append(text_line['date'])
							list_of_text.append(text_line['text'])
							list_of_window.append(mean_array)
							if text_line['CAT2'] == 'Message-NonUtilisable':
								list_of_label.append('Not_Crisis_period')
							else:
								if text_line['type_crisis'] == 'Flood' or text_line['type_crisis'] == 'Hurricane' or text_line['type_crisis'] == 'Storm':
									list_of_label.append('Ecological_crisis')
								else:
									list_of_label.append('Sudden_Crisis')

							end = 1
	df_return = pd.DataFrame({'Date': list_of_date, 'Text': list_of_text, 'Window': list_of_window, 'label': list_of_label})
	return df_return

