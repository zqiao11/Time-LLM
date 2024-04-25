# CNRS Multimodal Crisis Detection Project

Classification of Crises in the US and France based on Tweets and Meteorological Time Series Data.


## 1. Data

### 1.1. US data

⭐ Introduce US data in general. 

❓What are the practical crisis in it?  'Hurricane', 'WildFire', 'Flood'?

❓ How many tweets in total and per crisis? Please complete the following Table.

| Hurricane | WildFire | Flood |
|-----------|----------|-------|
|           |          |       |

❓ What are the classes/labels we want to classify? Predictable Crisis, Sudden Crisis?

❓ What is the frequency of the TS?


### 1.2. French data
⭐ Introduce French data in general. 

❓ How many tweets in total and per crisis? Please complete the table.

| Fire | Explosion | Attack | Collapse | Storms | Flood | Hurricane | Non-Crisis |
|------|-----------|--------|----------|--------|-------|-----------|------------|
|      |           |        |          |        |       |           |            |



3 high-level classes are considered: 
- Sudden Crisis ('Fire', 'Explosion', 'ATTACK', 'Collapse')
- Ecological Crisis ('Storms', 'Flood', 'Hurricane')
- Non-Crisis

Time series are collect in a frequency of 3 hours.


## 2. Creation of Multi-modal samples
To formulate multi-modal datasets, we need to link each tweet with its associated time series. After linking, each multi-modal sample in both US and French data is a dictionary with 4 fields:
- `'Date'`: `str`, date of the tweet.
- `'Text'`: `str`, text of the tweet.
- `'Window'`: `ndarray`, meteorological time series from 1 related station. Shape is (L, C).
- `'label'`: `str`, class label of crisis.


### 2.1 US data
⭐ Link based on date and location information.


### 2.2 French data

⭐ Link based on date and Event.