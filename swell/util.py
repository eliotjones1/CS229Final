import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

def create_swell_data(path):
    data = pd.read_csv(path)
    data = data.drop(columns=['Condition Label', 'NasaTLX class', 'condition'])
    return data


def run_rf(data, print_metrics=False, full_metrics=False, verbose=0):
    X = data.drop(columns=['NasaTLX Label', 'subject_id'], axis=1)
    y = data['NasaTLX Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    if len(X_train) < 10:
        return -1, 0, 0, 0
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42,
                                verbose=verbose)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)

    rf.fit(X_train_scaled, y_train)
    feature_importances = rf.feature_importances_

    features = X.columns
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    y_pred = rf.predict(X_test_scaled)

    if full_metrics:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
    else:
        precision, recall, f1 = 0, 0, 0
    accuracy = accuracy_score(y_test, y_pred)
    if print_metrics:
        print("CV Scores:", cv_scores)
        print("Feature ranking:")
        for f in range(X_train.shape[1]):
            print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
    return accuracy, precision, recall, f1

def run_xgb(data):
    X = data.drop(columns=['NasaTLX Label', 'subject_id'], axis=1)
    y = data['NasaTLX Label']

    # Model #1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #GridSearch CV
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.5, 0.7, 0.9],
        'colsample_bytree': [0.5, 0.7, 0.9],
    }

    # 10-fold CV
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

    xgb_reg = xgb.XGBRegressor(objective='binary:logistic',
                               eval_metric = 'logloss',
                               eta = 0.1,
                               subsample = 0.3)

    # Perform GridSearchCV with 10-fold cross-validation
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Print the best parameters and best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy score found: ", grid_search.best_score_)

    #Train XGBoost model
    best_params = grid_search.best_params_
    best_xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', **best_params)
    best_xgb_model.fit(X_train_scaled, y_train)

    #Predictions
    y_pred = best_xgb_model.predict(X_test_scaled)

    #Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on test set:", accuracy)
    return accuracy