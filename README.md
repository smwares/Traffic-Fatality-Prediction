# Traffic Fatality Predictions
Predicting traffic fatalities and exploring the dataset to identify specific contributing factors

# Table of Contents
1. [Introduction](#introduction)
2. [Objective](#objective)
3. [The Dataset](#the-dataset)
4. [Data Preparation](#data-preparation)
5. [Modeling](#modeling)
6. [Conclusion](#conclusion)
7. [Future Plans](#future-plans)
8. [References](#references)

## Introduction
According to the WHO, nearly 1.2 million people die worldwide every year from traffic accidents, with 20-50 million suffering injuries. Besides the cost of human lives and risk of injury, the cost of damages can be in the millions and be as much as 3% of a country's GDP. In an ideal situation with unlimited resources and manpower, all of these can be tended to, but unfortunately with increasing restraints on budgeting it's now more important than ever to plan appropriately to maximize on saving lives and costs. This is an area that data science can help make a difference in through observing trends and making predictions for better future planning.

## Objective
- To build models that can predict whether or not a fatality will occur in a vehicle, if an accident will lead to multiple fatalities, and to predict the number of fatalities from an accident
- Explore key factors in the dataset that may give further insight and possibly answer other questions

## The Dataset
- National Highway Traffic System Administration (NHTSA) traffic data from 2015 and 2016
- Dataset was part of the BigQuery library
- 40 tables, but only 4 were used as they contained the most amount of features (and there is lots of overlap between the tables)
- Accident table has entries of every fatal accident in the USA
- Vehicle table has entries for every vehicle that was involved in a fatal accident in the USA
- Accident table contains 70 features, while vehicle table contains 115
- While every entry in the accident table has at least one fatality, roughly half the vehicles in the vehicle table have fatalities since not every car involved in an accident would necessarily have a fatality inside

## EDA

[Link to bar graphs for each feature](./src/EDA_formatting.ipynb#bar_graph)

## Data Preparation
- Vehicle and Accident tables for 2015 and 2016 and the descriptions for each feature of these tables were saved using Jupyter Notebooks on Kaggle (link to code: https://www.kaggle.com/code/smwares/traffic-fatality-prediction )
- Vehicle and Accident tables were joined based on the consecutive number column, which is a unique number assigned to each accident
- After doing EDA on the tables and reading through the features for each feature, several features were removed due to irrelevance or having too many null entries
- Rows of data that contained unknown or not reported data were deleted
- Categorical features were one-hot encoded
- Numerical features were scaled
- Date/time features were converted to cyclical

## Modeling

Principal component analyis, recursive feature elimination with cross-validation, XGBoost's feature importance and variance threshold were all used to determine the optimal number of features to be used (PCA was only used for the first model). Random search cross-validation was used in combination with XGBoost's classification and regression to build models with best evaluation metrics. For the second and third types of models, one variant of each were built using the Accident table only and another of each were built using the combined Accident and Vehicle table.

### First Model Type

First model that was built was a model to predict if a vehicle involved in an accident had anyone who suffered a fatality.

PCA:
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-pca.png)
According to the PCA, the top 15 features explains nearly half (47%) of the variance in the data.

RFE-CV:
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-rfecv.png)
The results of RFE-CV also confirms that 15 features is optimal for making predictions.

Using variance threshold with a threshold of 0.2, 37 features are sufficient to meet the threshold.
Correlation matrix of top features using a variance threshold of 0.2:
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-corr-matrix-heatmap-vt.png)

XGBoost model results using top features determined by variance threshold of 0.2 (2015 test set):
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-classification-report-vt.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-cm-vt.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-auc-roc-curve-vt.png)

XGBoost feature importance:
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-xgb-feat-importance.png)

Correlation matrix of top 15 features determined by XGBoost:
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-corr-matrix-heatmap.png)

XGBoost stock results using top 15 features (2015 test set):
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-classification-report-def.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-cm-def.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-auc-roc-curve-def.png)

XGBoost stock results using all features (2015 test set):
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-classification-report-all.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-cm-all.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-auc-roc-curve-all.png)

XGBoost stock results using all features (2016 set):
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2016-classification-report-all.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2016-cm-all.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2016-auc-roc-curve-all.png)

XGBoost random search CV results using top 15 features (2015 test set):
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-classification-report.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-cm.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2015test-auc-roc-curve.png)

XGBoost random search CV results using top 15 features (2016 set):
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2016-classification-report.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2016-cm.png)
![alt text](https://raw.githubusercontent.com/smwares/Traffic-Fatality-Prediction/main/img/model1-2016-auc-roc-curve.png)

Building a model using XGBoost classification with random search cross-validation and using top 15 features according to XGBoost's feature importance results in a model that can predict a fatality in a vehicle with 80% accuracy, even when using 2016's data. For reference, using all of the features results in a model with 88% accuracy. Comparing evaluation metrics of models that were built using different methods of determining feature importance, it seems that XGBoost's feature importance slightly edges out variance threshold, despite using only 15 features VS 37 features that variance threshold determined to be best for meeting a threshold of 0.2. For the rest of the models, XGBoost's feature importance will be used.

### Second Model Type - Accident Dataset Only

### Second Model Type - Accident and Vehicle Datasets Combined

Second model that was built was a model to predict if an accident resulted in a single fatality or multiple fatalities, and the third model that was built was a model to predict the number of fatalities in an accident. Despite trying different amounts of data and random search to find the best parameters, neither of these models provided good results as there isn't enough data, especially for the second model since an overwhelming amount of accidents in the dataset contained a single fatality.

### Third Model Type - Accident Dataset Only

### Third Model Type - Accident and Vehicle Datasets Combined


For additional component analyses and evaluation results, please refer to the modeling notebook files.

## Conclusion

- Predictions can be reliably made based on the dataset provided by the NHTSA
- Such predictions can help with studying trends and statistics, which can be useful for insurance companies
- This dataset can also be used as part of larger datasets that can also be used to answer various questions and explore multiple topics
- Topics could include traffic predictions when combined with dataset that has information on ALL accidents, EMS response times and correlations to fatalities when combined with a more complete EMS dataset

## Future Plans
 
- Get data on Canadian statistics for relevancy
- Get data beyond two years for both a larger training dataset and a testing dataset
- Get information on ALL accidents to explore more model building options other than predicting fatalities

## References
<a href="https://www.kaggle.com/datasets/usdot/nhtsa-traffic-fatalities">USA Traffic Fatality Records</a> - Kaggle

<a href="https://alexionpartners.com/homepage/aerial-view-of-road-interchange-or-highway-intersection-with-busy-urban-traffic-speeding-on-the-road-junction-network-of-transportation-taken-by-drone">Aerial view of road interchange or highway intersection with busy urban traffic speeding on the road. Junction network of transportation taken by drone</a> - Alexion Partners

<a href="https://paintingvalley.com/download-image#traffic-light-drawing-34.jpg">Traffic Light Drawing</a> - Painting Valley
