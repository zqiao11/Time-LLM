# CNRS Multimodal Crisis Detection Project

Classification of Crises in the US and France based on Tweets and Meteorological Time Series Data.


## 1. Data

### 1.1. US data

⭐ Introduce US data in general. 

US Data is the dataset from Humaid from Alam, F., Qazi, U., Imran, M., & Ofli, F. (2021, May). Humaid: Human-annotated disaster incidents data from twitter with deep learning benchmarks. In Proceedings of the International AAAI Conference on Web and social media (Vol. 15, pp. 933-942). but to to link nlp data and time series i needed to find a way. Thankfully i saw that the same authors made another paper Suwaileh, R., Elsayed, T., & Imran, M. (2023). IDRISI-RE: A generalizable dataset with benchmarks for location mention recognition on disaster tweets. Information Processing & Management, 60(3), 103340. This paper introduce IDRISI, a subset of HumAID with location mention. So i used Idrisi and i linked all the data wich are related to US crisis because i got acess to american meterological data.

❓What are the practical crisis in it?  'Hurricane', 'WildFire', 'Flood'?

Exactly

❓ How many tweets in total and per crisis? Please complete the following Table.

|   Crisis   | Hurricane | WildFire | Flood | Earthquake | Total |
|------------|-----------|----------|-------|------------|-------|
|  By Crisis |   4120    |   1087   |  431  |    1105    |  6743 |
|   Crisis   |   3941    |   1087   |  431  |     896    |  6355 |
| Not Crisis |    179    |     0    |   0   |     209    |   388 |   

❓ What are the classes/labels we want to classify? Predictable Crisis, Sudden Crisis?

The Labels are Not_Crisis_period ,Sudden_Crisis, Predictible_Crisis


❓ What is the frequency of the TS?
The time series is a daily summary of the meterogical information so the frequency is daily.

### 1.2. French data
⭐ Introduce French data in general. 

❓ How many tweets in total and per crisis? Please complete the table.

|   Crisis   | Fire | Explosion | Attack | Collapse | Storms | Flood | Hurricane |    Total   |
|------------|------|-----------|--------|----------|--------|-------|-----------|------------|
|  By Crisis | 1040 |    1890   |  1382  |   1269   |  5760  |  3593 |    2160   |    18996   |
|------------|------|-----------|--------|----------|--------|-------|-----------|------------|
|   Crisis   | 1464 |    1747   |  1208  |   322    |  1092  |  1426 |    742    |    8001    |
| Not Crisis | 1478 |    143    |   174  |   947    |  4668  |  2167 |    1418   |    10995   |



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

Done

### 2.2 French data

⭐ Link based on date and Event.

Done
