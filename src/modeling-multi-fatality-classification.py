import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# import the formatted dataset
df = pd.read_csv('../data/accident_vehicle_2015_formatted.csv')

# add in a new column that takes multiple fatalities into account, True if more than one, False if one
df['multi-fatality'] = False
df.loc[(df['number_of_fatalities'] > 1), 'multi-fatality'] = True

# create a function to print the confusion matrix
def print_cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    
    sns.set_context('talk')
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

# create a function to plot the ROC curve and print the AUC
def print_roc_auc(y_test, y_pred_proba):
    fpr, tpr, thresh = roc_curve(y_test, y_pred_proba, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    sns.set_context('talk')
    sns.set_style('darkgrid')
    plt.figure(figsize=(12,8))
    plt.plot(fpr, tpr, label=str(roc_auc))
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

# create a function to print a correlation matrix heat map
def print_corr(df, annotate=False):
    df_corr = df.corr(method='kendall', numeric_only=True)

    # display a heatmap of the correlation matrix
    sns.set(rc={"figure.figsize":(20, 16)})
    sns.heatmap(df_corr, annot=annotate, annot_kws={'size': 12}, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()

# scale the numerical features
cols_to_rescale = ['number_of_motor_vehicles_in_transport_mvit',
                   'number_of_occupants',
                   'day_of_crash',
                   'total_lanes_in_roadway',
                   'number_of_forms_submitted_for_persons_not_in_motor_vehicles',
                   'number_of_persons_not_in_motor_vehicles_in_transport_mvit',
                   'number_of_persons_in_motor_vehicles_in_transport_mvit',
                   'latitude',
                   'longitude',
                   'light_condition',
                   'number_of_drunk_drivers']

scaler = StandardScaler()
scaler.fit(df[cols_to_rescale])
rescaled_cols = scaler.transform(df[cols_to_rescale])

# create a separate dataframe
df_scaled = df.drop((['fatalities_in_vehicle', 'number_of_fatalities', 'multi-fatality']), axis=1).copy()
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
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df['multi-fatality'], test_size=0.2, random_state=21)

xgbc_model = XGBClassifier(objective='binary:logistic')

rfe_cv = RFECV(estimator=xgbc_model,
               step=50,
               cv=StratifiedKFold(3),
               scoring='accuracy',
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

# Seems like 10 features gives the optimal amount of accuracy

# Model using XGB feature importance:

# use XGBoost and feature importance to get the most important features
xgb_all = XGBClassifier(objective='binary:logistic', n_jobs=-1)
xgb_all.fit(X_train, y_train)

# use the plot_importance function to plot the most important features
fig, ax = plt.subplots(figsize=(20,200))
plot_importance(xgb_all, ax=ax)
plt.show()

# save predictions from the test set
y_pred = xgb_all.predict(X_test)
y_pred_proba = xgb_all.predict_proba(X_test)

# evaluate the results
print('Classification report\n', classification_report(y_test, y_pred))
print_cm(y_test, y_pred)

# plot the ROC curve and print the AUC
print_roc_auc(y_test, y_pred_proba[:, 1])

# get a dict of the features sorted by feature importance
xgb_all_booster = xgb_all.get_booster()
importance_dict = xgb_all_booster.get_score(importance_type='weight')
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
xgb_feats = [feature[0] for feature in sorted_importance]

# save the 15 most important features
X = df_scaled[xgb_feats[:15]]
y_class = df['multi-fatality']
y_reg = df['number_of_fatalities']

# create another train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=21)

# print correlation matrix heat map
print_corr(pd.concat([X, y_class, df['fatalities_in_vehicle'], y_reg], axis=1), annotate=True)

# fit a model
xgb_model = XGBClassifier(objective='binary:logistic', n_jobs=-1)
xgb_model.fit(X_train, y_train)

# save predictions from the test set
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# evaluate the results
print('Classification report\n', classification_report(y_test, y_pred))
print_cm(y_test, y_pred)

# plot the ROC curve and print the AUC
print_roc_auc(y_test, y_pred_proba[:, 1])

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
print_corr(pd.concat([X_vt, y_class, df['fatalities_in_vehicle'], y_reg], axis=1), annotate=False)

# create another train-test split based on the variance threshold
X_train, X_test, y_train, y_test = train_test_split(X_vt, y_class, test_size=0.2, random_state=21)

# fit a model
xgb_model = XGBClassifier(objective='binary:logistic', n_jobs=-1)
xgb_model.fit(X_train, y_train)

# save predictions from the test set
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# evaluate the results
print('Classification report\n', classification_report(y_test, y_pred))
print_cm(y_test, y_pred)

# plot the ROC curve and print the AUC
print_roc_auc(y_test, y_pred_proba[:, 1])

# Model using random search and top 15 most important features

# create another train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=21)

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
          'n_jobs':[-1],
          'objective':['binary:logistic']}

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
          'n_jobs':[-1],
          'objective':['binary:logistic']}

# suppress future warnings to save on CPU cycles
warnings.filterwarnings('ignore', category=FutureWarning)

# initialize a second XGB classifier for grid search and do the grid search
xgb_model2 = XGBClassifier()

rs = RandomizedSearchCV(estimator=xgb_model2, param_distributions=params, cv=5, scoring='recall', n_iter=25, verbose=3, random_state=21)
rs.fit(X_train, y_train)

# get the best parameters and the best score
print('Best score:', rs.best_score_)
print('Best parameters:', rs.best_params_)
best_xgb_params = rs.best_params_

# build a new model using the best parameters
xgb_best_model = XGBClassifier(**best_xgb_params)
xgb_best_model.fit(X_train, y_train)

# make new predictions
y_best_pred = xgb_best_model.predict(X_test)
y_best_pred_proba = xgb_best_model.predict_proba(X_test)

# evaluate this new model
print('Classification report\n', classification_report(y_test, y_best_pred))
print_cm(y_test, y_best_pred)
print_roc_auc(y_test, y_best_pred_proba[:, 1])

# Evaluation using 2016 data:

# import the 2016 formatted data for testing
df_2016 = pd.read_csv('../data/accident_vehicle_2016_formatted.csv')

# add in a new column that takes multiple fatalities into account, True if more than one, False if one
df_2016['multi-fatality'] = False
df_2016.loc[(df_2016['number_of_fatalities'] > 1), 'multi-fatality'] = True

rescaled_cols_2016 = scaler.transform(df_2016[cols_to_rescale])

# create a separate dataframe
df_2016_scaled = df_2016.drop((['fatalities_in_vehicle', 'number_of_fatalities', 'multi-fatality']), axis=1).copy()
df_2016_scaled[cols_to_rescale] = rescaled_cols_2016.astype('float32')

# cosine transform the month, day of week and hour features since they're cyclical
df_2016_scaled['month_of_crash'] = np.cos(2 * np.pi * df_2016_scaled['month_of_crash'] / 12)
df_2016_scaled['day_of_week'] = np.cos(2 * np.pi * df_2016_scaled['day_of_week'] / 7)
df_2016_scaled['hour_of_crash'] = np.cos(2 * np.pi * df_2016_scaled['hour_of_crash'] / 24)

df_2016_scaled['month_of_crash'] = df_2016_scaled['month_of_crash'].astype('float32')
df_2016_scaled['day_of_week'] = df_2016_scaled['day_of_week'].astype('float32')
df_2016_scaled['hour_of_crash'] = df_2016_scaled['hour_of_crash'].astype('float32')

X_16 = df_2016_scaled[xgb_feats[:15]]
y_class_16 = df_2016['multi-fatality']

# make new predictions
y_pred_16 = xgb_best_model.predict(X_16)
y_pred_proba_16 = xgb_best_model.predict_proba(X_16)

# evaluate this new model
print('Classification report\n', classification_report(y_class_16, y_pred_16))
print_cm(y_class_16, y_pred_16)
print_roc_auc(y_class_16, y_pred_proba_16[:, 1])

#create a separate dataframe to evaluate XGB model that uses all features
df_2016_scaled_all = df_2016_scaled.copy()

for col in df_scaled.columns:
    if col not in df_2016_scaled_all.columns:
        df_2016_scaled_all[col] = 0

#evaluate model that was built using all features
y_pred_all_16 = xgb_all.predict(df_2016_scaled_all[df_scaled.columns])
y_pred_all_proba_16 = xgb_all.predict_proba(df_2016_scaled_all[df_scaled.columns])

print('Classification report\n', classification_report(y_class_16, y_pred_all_16))
print_cm(y_class_16, y_pred_all_16)
print_roc_auc(y_class_16, y_pred_all_proba_16[:, 1])
