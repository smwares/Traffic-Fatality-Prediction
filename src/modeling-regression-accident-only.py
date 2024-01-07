import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# import the formatted dataset
df = pd.read_csv('../data/accident_2015_formatted.csv')

df['number_of_fatalities'].value_counts()

df['number_of_fatalities'].hist(bins=10)

# create a function to print errors
def print_errors(y_test, y_pred):
    print('Mean absolute error:', mean_absolute_error(y_test, y_pred))
    print('Mean squared error:', mean_squared_error(y_test, y_pred))
    print('Root mean squared error:', np.sqrt(mean_squared_error(y_test, y_pred)))
    print('R-squared error:', r2_score(y_test, y_pred))

# create a function to print a correlation matrix heat map
def print_corr(df, annotate=False):
    df_corr = df.corr(method='kendall', numeric_only=True)

    # display a heatmap of the correlation matrix
    sns.set(rc={"figure.figsize":(20, 16)})
    sns.heatmap(df_corr, annot=annotate, annot_kws={'size': 12}, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()

# scale the numerical features
cols_to_rescale = ['number_of_parked_working_vehicles',
                   'number_of_forms_submitted_for_persons_not_in_motor_vehicles',
                   'number_of_persons_not_in_motor_vehicles_in_transport_mvit',
                   'number_of_persons_in_motor_vehicles_in_transport_mvit',
                   'day_of_crash',
                   'latitude',
                   'longitude',
                   'light_condition',
                   'number_of_drunk_drivers']

scaler = StandardScaler()
scaler.fit(df[cols_to_rescale])
rescaled_cols = scaler.transform(df[cols_to_rescale])

# create a separate dataframe
df_scaled = df.drop(('number_of_fatalities'), axis=1).copy()
df_scaled[cols_to_rescale] = rescaled_cols.astype('float32')

# cosine transform the month, day of week and hour features since they're cyclical
df_scaled['month_of_crash'] = np.cos(2 * np.pi * df_scaled['month_of_crash'] / 12)
df_scaled['day_of_week'] = np.cos(2 * np.pi * df_scaled['day_of_week'] / 7)
df_scaled['hour_of_crash'] = np.cos(2 * np.pi * df_scaled['hour_of_crash'] / 24)

df_scaled['month_of_crash'] = df_scaled['month_of_crash'].astype('float32')
df_scaled['day_of_week'] = df_scaled['day_of_week'].astype('float32')
df_scaled['hour_of_crash'] = df_scaled['hour_of_crash'].astype('float32')

# Model using RFE, CV and XGBClassifier:

# use recursive feature elimination with cross-validation to get a ranking of the features in terms of importance
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['number_of_fatalities'], test_size=0.2, random_state=21)

xgbr_model = xgb.XGBRegressor(objective='count:poisson')

rfe_cv = RFECV(estimator=xgbr_model,
               step=10,
               cv=3,
               scoring='neg_mean_squared_error',
               n_jobs=-1)

rfe_cv.fit(X_train, y_train)

# print the number of optimal features that RFE found
print('Optimal number of features:', rfe_cv.n_features_)

X_train.columns[rfe_cv.support_]

# print the cross-validation scores for each level of features being used
cv_scores = rfe_cv.cv_results_['mean_test_score']
cv_scores_std = rfe_cv.cv_results_['std_test_score']
number_of_features = np.arange(1, len(cv_scores) + 1, 1)

sns.set_style('darkgrid')
fig, ax = plt.subplots(figsize=(20,10), layout='constrained')
fig.suptitle("RFECV - Optimal number of features vs. Score")
plt.xlabel("# of Features")
ax.set_xticks([i for i in number_of_features])
plt.ylabel("Mean Accuracy")
plt.plot(number_of_features, cv_scores, 'o-')
plt.fill_between(number_of_features, cv_scores - cv_scores_std, cv_scores + cv_scores_std, alpha=0.25)
plt.show()

# Seems like 19 features gives the optimal amount of accuracy

# Model using XGB feature importance:

# use XGBoost and feature importance to get the most important features
xgb_all = xgb.XGBRegressor(objective='count:poisson')
xgb_all.fit(X_train, y_train)

# use the plot_importance function to plot the most important features
fig, ax = plt.subplots(figsize=(20,200))
xgb.plot_importance(xgb_all, ax=ax)
plt.show()

# save predictions from the test set
y_pred = xgb_all.predict(X_test)

# evaluate the results
print_errors(y_test, y_pred)

# get a dict of the features sorted by feature importance
xgb_all_booster = xgb_all.get_booster()
importance_dict = xgb_all_booster.get_score(importance_type='weight')
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
xgb_feats = [feature[0] for feature in sorted_importance]

# save the 16 most important features
X = df_scaled[xgb_feats[:16]]
y = df['number_of_fatalities']

# print correlation matrix heat map
print_corr(pd.concat([X, y], axis=1), annotate=True)

# due to the extremely high positive correlation between two features, drop one
X.drop('number_of_forms_submitted_for_persons_not_in_motor_vehicles', axis=1, inplace=True)

# create another train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# fit a model
xgb_model = xgb.XGBRegressor(objective='count:poisson')
xgb_model.fit(X_train, y_train)

# save predictions from the test set
y_pred = xgb_model.predict(X_test)

# evaluate the results
print_errors(y_test, y_pred)

# Model using variance threshold:

# use variance threshold to get a set of features that contribute the most to the variance of data
vt = VarianceThreshold(0.20)
df_vt = vt.fit_transform(df_scaled)

# save the features that met the threshold
important_features = df_scaled.columns[vt.get_support()]
print('Features:\n', important_features, '\nNumber of features:', len(important_features))

# save X and y based on the features that met the threshold of the Variance Threshold
X_vt = df_scaled[important_features]

# print correlation matrix heat map
print_corr(pd.concat([X_vt,  y], axis=1), annotate=True)

# create another train-test split based on the variance threshold
X_train, X_test, y_train, y_test = train_test_split(X_vt, y, test_size=0.2, random_state=21)

# fit a model
xgb_model = xgb.XGBRegressor(objective='count:poisson')
xgb_model.fit(X_train, y_train)

# save predictions from the test set
y_pred = xgb_model.predict(X_test)

# evaluate the results
print_errors(y_test, y_pred)

# Model using random search and top 15 most important features

# create another train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# use random search to improve the model while simulatenously mitigating overfitting

# set up the parameter grid
params = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001],
          'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
          'gamma': [0, .01, 0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5],
          'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
          'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
          'colsample_bytree': [0.2, 0.4, 0.6, 0.8, 1.0],
          #'colsample_bylevel': [0.2, 0.4, 0.6, 0.8, 1.0],
          #'colsample_bynode': [0.2, 0.4, 0.6, 0.8, 1.0],
          'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
          'reg_lambda': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],
          'objective':['count:poisson']}

# 2nd set of parameters just in case the first set is too compute and/or memory heavy
params2 = {#'max_depth': [2, 5, 8],
          'learning_rate': [0.1, 0.01, 0.001],
          'n_estimators': [100, 500, 1000],
          'gamma': [0.01, 0.1],
          'min_child_weight': [1, 4, 7],
          'subsample': [0.5, 1.0],
          'colsample_bytree': [0.5, 1.0],
          #'colsample_bylevel': [0.2, 0.4, 0.6, 0.8, 1.0],
          #'colsample_bynode': [0.2, 0.4, 0.6, 0.8, 1.0],
          'reg_alpha': [0, 0.5, 1],
          'reg_lambda': [0, 0.5, 1],
          'objective':['count:poisson']}

# suppress future warnings to save on CPU cycles
warnings.filterwarnings('ignore', category=FutureWarning)

# initialize a second XGB classifier for grid search and do the grid search
xgb_model2 = xgb.XGBRegressor()

rs = RandomizedSearchCV(estimator=xgb_model2, param_distributions=params, cv=5, scoring='neg_mean_squared_error', n_iter=25, verbose=3, random_state=21)
rs.fit(X_train, y_train)

# get the best parameters and the best score
print('Best score:', rs.best_score_)
print('Best parameters:', rs.best_params_)
best_xgb_params = rs.best_params_

# build a new model using the best parameters
xgb_best_model = xgb.XGBRegressor(**best_xgb_params)
xgb_best_model.fit(X_train, y_train)

# make new predictions
y_best_pred = xgb_best_model.predict(X_test)

# evaluate this new model
print_errors(y_test, y_best_pred)

# Evaluation using 2016 data:

# import the 2016 formatted data for testing
df_2016 = pd.read_csv('../data/accident_2016_formatted.csv')

rescaled_cols_2016 = scaler.transform(df_2016[cols_to_rescale])

# create a separate dataframe
df_2016_scaled = df_2016.drop(('number_of_fatalities'), axis=1).copy()
df_2016_scaled[cols_to_rescale] = rescaled_cols_2016.astype('float32')

# cosine transform the month, day of week and hour features since they're cyclical
df_2016_scaled['month_of_crash'] = np.cos(2 * np.pi * df_2016_scaled['month_of_crash'] / 12)
df_2016_scaled['day_of_week'] = np.cos(2 * np.pi * df_2016_scaled['day_of_week'] / 7)
df_2016_scaled['hour_of_crash'] = np.cos(2 * np.pi * df_2016_scaled['hour_of_crash'] / 24)

df_2016_scaled['month_of_crash'] = df_2016_scaled['month_of_crash'].astype('float32')
df_2016_scaled['day_of_week'] = df_2016_scaled['day_of_week'].astype('float32')
df_2016_scaled['hour_of_crash'] = df_2016_scaled['hour_of_crash'].astype('float32')

#create a separate dataframe to evaluate XGB model that uses all features
df_2016_scaled_all = df_2016_scaled.copy()

for col in df_scaled.columns:
    if col not in df_2016_scaled_all.columns:
        df_2016_scaled_all[col] = 0

X_16 = df_2016_scaled_all[X.columns]
y_16 = df_2016['number_of_fatalities']

# make new predictions
y_pred_16 = xgb_best_model.predict(X_16)

# evaluate this new model
print_errors(y_16, y_pred_16)

#evaluate model that was built using all features
y_pred_all_16 = xgb_all.predict(df_2016_scaled_all[df_scaled.columns])

# evaluate this new model
print_errors(y_16, y_pred_all_16)

# Build a model using data from 2015 and 2016

df_combined = pd.concat([df, df_2016], axis=0)
y_combined = pd.concat([y, y_16], axis=0)

# check which features have null values
print(df_combined.columns[(df_combined.isna().sum() > 0).tolist()])

# fill the null values with false
df_combined.fillna(value=False, inplace=True)

# rescale the data
scaler = StandardScaler()
scaler.fit(df_combined[cols_to_rescale])
rescaled_cols = scaler.transform(df_combined[cols_to_rescale])

# create a separate dataframe
df_combined_scaled = df_combined.drop(('number_of_fatalities'), axis=1).copy()
df_combined_scaled[cols_to_rescale] = rescaled_cols.astype('float32')

# cosine transform the month, day of week and hour features since they're cyclical
df_combined_scaled['month_of_crash'] = np.cos(2 * np.pi * df_combined_scaled['month_of_crash'] / 12)
df_combined_scaled['day_of_week'] = np.cos(2 * np.pi * df_combined_scaled['day_of_week'] / 7)
df_combined_scaled['hour_of_crash'] = np.cos(2 * np.pi * df_combined_scaled['hour_of_crash'] / 24)

df_combined_scaled['month_of_crash'] = df_combined_scaled['month_of_crash'].astype('float32')
df_combined_scaled['day_of_week'] = df_combined_scaled['day_of_week'].astype('float32')
df_combined_scaled['hour_of_crash'] = df_combined_scaled['hour_of_crash'].astype('float32')

# create a train test split
X_train, X_test, y_train, y_test = train_test_split(df_combined_scaled[X.columns], y_combined, test_size=0.2, random_state=21)

# use random search to find the best model
xgb_model3 = xgb.XGBRegressor()

rs2 = RandomizedSearchCV(estimator=xgb_model3, param_distributions=params, cv=5, scoring='neg_mean_squared_error', n_iter=25, verbose=3, random_state=21)
rs2.fit(X_train, y_train)

# get the best parameters and the best score
print('Best score:', rs2.best_score_)
print('Best parameters:', rs2.best_params_)
best_xgb_params2 = rs2.best_params_

# build a new model using the best parameters
xgb_best_model2 = xgb.XGBRegressor(**best_xgb_params2)
xgb_best_model2.fit(X_train, y_train)

# make new predictions
y_best_pred2 = xgb_best_model2.predict(X_test)

# evaluate this new model
print_errors(y_test, y_best_pred2)
