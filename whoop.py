import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


'''
Columns: 
'ScrSubjectID', 'Days from  Round1 Day1', 'Round 1 Exercise',
       'Pre PANAS Positive Affect', 'Pre PANAS Negative Affect',
       'Post PANAS Positive Affect', 'Post PANAS Negative Affect',
       'Pre STAI State Anxiety', 'Post STAI State Anxiety',
       'BW Exercise Minutes (timer)', 'Did you complete BW exercises?',
       'Sleep Q Score', 'Full STAI State Anxiety', 'Full STAI Trait Anxiety',
       'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency',
       'Respiration Rate', 'Sleep T Score',
       'Overall, how easy was it to use the instructions/videos for doing the intervention?',
       'Overall, how easy was it do do the daily intervention?',
       'subjective experience: grounding', 'subjective experience: focusing',
       'subjective experience: calming', 'subjective experience: energizing',
       'subjective experience: brief and simple',
       'subjective experience: other positive',
       'subjective experience: some challenges'
       
       Want to keep:
       ['ScrSubjectID', 'Days from  Round1 Day1', 'Round 1 Exercise',
       'Pre PANAS Positive Affect', 'Pre PANAS Negative Affect',
       'Post PANAS Positive Affect', 'Post PANAS Negative Affect',
       'Pre STAI State Anxiety', 'Post STAI State Anxiety',
       'BW Exercise Minutes (timer)', 'Did you complete BW exercises?',
       'Sleep Q Score', 'Full STAI State Anxiety', 'Full STAI Trait Anxiety',
       'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency',
       'Respiration Rate', 'Sleep T Score']
'''
if __name__ == '__main__':
    data = pd.read_excel('whoop/BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')
    
    # List of columns we want to keep
    list_of_columns = ['ScrSubjectID', 'Days from  Round1 Day1', 'Round 1 Exercise',
                       'Pre PANAS Positive Affect', 'Pre PANAS Negative Affect',
                       'Post PANAS Positive Affect', 'Post PANAS Negative Affect',
                       'Pre STAI State Anxiety', 'Post STAI State Anxiety',
                       'BW Exercise Minutes (timer)', 'Did you complete BW exercises?',
                       'Sleep Q Score', 'Full STAI State Anxiety', 'Full STAI Trait Anxiety',
                       'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency',
                       'Respiration Rate', 'Sleep T Score']
    
    whoop_dataset = data[list_of_columns]
    whoop_dataset = whoop_dataset[whoop_dataset['Days from  Round1 Day1'] >= 0]
    # One-hot encoding of Round 1 Exercises
    whoop_dataset = pd.get_dummies(whoop_dataset, columns=["Round 1 Exercise"])
    
    # Replace entries with '.' with 'NaN'
    whoop_dataset = whoop_dataset.replace('.', np.nan)
    
    # Remove columns that are mostly empty
    percent_missing = whoop_dataset.isnull().mean() * 100
    print(percent_missing)
    threshold = 50
    columns_to_drop = percent_missing[percent_missing > threshold].index.tolist()
    whoop_dataset_cleaned = whoop_dataset.drop(columns=columns_to_drop)
    
    # Check completeness
    complete_rows = whoop_dataset_cleaned.dropna().shape[0]
    print(f"Number of rows with full information: {complete_rows}")
    
    # Drop incomplete rows
    complete_whoop_dataset = whoop_dataset_cleaned.dropna()
    
    # Check completeness per person
    grouped_counts = complete_whoop_dataset.groupby('ScrSubjectID').apply(lambda group: group.dropna().shape[0])
    
    # Want to test on Pre STAI State Anxiety. If it is above 38, say "stressed" (1), else "not stressed" (0). 
    complete_whoop_dataset.loc[:, 'Pre STAI State Anxiety'] = np.where(complete_whoop_dataset['Pre STAI State Anxiety'] > 38, 1, 0)
    print(complete_whoop_dataset.columns)
    
    # ######## TRIAL ONE: Using full dataset, without focusing on SubjectID ########
    # print('TRIAL ONE: Full dataset')
    # trial_one_data = complete_whoop_dataset
    # trial_one_data = trial_one_data.drop(columns=['ScrSubjectID', 'Post STAI State Anxiety'])
    
    # X = trial_one_data.drop('Pre STAI State Anxiety', axis=1)  
    # y = trial_one_data['Pre STAI State Anxiety']
    
    # # Model #1: RandomForest
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # # Hyperparameters according to the paper
    # rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42)
    
    # # 10-fold CV
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)
    
    # # Fit the model
    # rf.fit(X_train_scaled, y_train)
    # feature_importances = rf.feature_importances_
    
    # print("CV Scores:", cv_scores)
    # features = X.columns
    # importances = rf.feature_importances_
    # indices = np.argsort(importances)[::-1]
    
    # print("Feature ranking:")
    # for f in range(X_train.shape[1]):
    #     print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")
        
    # # Evaluate the model
    # y_pred = rf.predict(X_test_scaled)
    
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='binary')
    # recall = recall_score(y_test, y_pred, average='binary')
    # f1 = f1_score(y_test, y_pred, average='binary')
    
    # # Print metrics
    
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    
    # ## SMALL MODEL, RHR, HRV, Sleep Score, Hours of Sleep, Sleep Efficiency, Respiration Rate ##
    # print('TRIAL TWO: Only Whoop data as features')
    # small_trial_data = complete_whoop_dataset
    # small_trial_data = small_trial_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency', 'Respiration Rate']]

    # X = small_trial_data.drop('Pre STAI State Anxiety', axis=1)  
    # y = small_trial_data['Pre STAI State Anxiety']

    # # Model #1: RandomForest
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # Hyperparameters according to the paper
    # rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42)

    # # 10-fold CV
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)

    # # Fit the model
    # rf.fit(X_train_scaled, y_train)
    # feature_importances = rf.feature_importances_

    # print("CV Scores:", cv_scores)
    # features = X.columns
    # importances = rf.feature_importances_
    # indices = np.argsort(importances)[::-1]

    # print("Feature ranking:")
    # for f in range(X_train.shape[1]):
    #     print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")

    # # Evaluate the model
    # y_pred = rf.predict(X_test_scaled)

    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='binary')
    # recall = recall_score(y_test, y_pred, average='binary')
    # f1 = f1_score(y_test, y_pred, average='binary')

    # # Print metrics

    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")

    #  ## LIMITED MODEL, RHR, HRV ##
    # print('TRIAL THREE: Only RHR and HRV features')
    # limited_trial_data = complete_whoop_dataset
    # limited_trial_data = limited_trial_data[['Pre STAI State Anxiety', 'RHR', 'HRV']]

    # X = limited_trial_data.drop('Pre STAI State Anxiety', axis=1)  
    # y = limited_trial_data['Pre STAI State Anxiety']

    # # Model #1: RandomForest
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # # Hyperparameters according to the paper
    # rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42)

    # # 10-fold CV
    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)

    # # Fit the model
    # rf.fit(X_train_scaled, y_train)
    # feature_importances = rf.feature_importances_

    # print("CV Scores:", cv_scores)
    # features = X.columns
    # importances = rf.feature_importances_
    # indices = np.argsort(importances)[::-1]

    # print("Feature ranking:")
    # for f in range(X_train.shape[1]):
    #     print(f"{f + 1}. feature {features[indices[f]]} ({importances[indices[f]]})")

    # # Evaluate the model
    # y_pred = rf.predict(X_test_scaled)

    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='binary')
    # recall = recall_score(y_test, y_pred, average='binary')
    # f1 = f1_score(y_test, y_pred, average='binary')

    # # Print metrics

    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    
#     ######## TRIAL FOUR: Using full dataset, person-specific forest plots########
#     print('TRIAL FOUR: Person-specific random forest ')
#     rf_data = complete_whoop_dataset
#     rf1_data = rf_data.drop(columns=['Post STAI State Anxiety'])

#     persons_data = rf1_data['ScrSubjectID'].unique()

#     accuracies = {}
#     list_of_accuracies = []
#     list_of_MSEs = []
#     for person in persons_data:
#         person_data = rf_data[rf_data['ScrSubjectID'] == person]
#         if len(person_data)<10:
#              continue
#         X = person_data.drop('Pre STAI State Anxiety', axis=1)  
#         y = person_data['Pre STAI State Anxiety']
    
#         # Model #1: Ridge Regression
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
    
#         # Hyperparameters according to the paper
#         rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42, verbose=1)
        
#         # 10-fold CV
#         # n_splits = len(person_data) if len(person_data) < 10 else 10
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)
        
#         # Fit the model
#         rf.fit(X_train_scaled, y_train)
#         feature_importances = rf.feature_importances_
        
#         # Evaluate the model
#         y_pred = rf.predict(X_test_scaled)

#         accuracy = accuracy_score(y_test, y_pred)
#         MSE = mean_squared_error(y_test, y_pred)


#         # Save accuracy
#         accuracies[person] = accuracy
#         list_of_accuracies.append(accuracy)
#         list_of_MSEs.append(MSE)

#     # for person, accuracy in accuracies.items():
#     #         print(f"Person {person}: accuracy = {accuracy}")

#     average_acc = np.mean(list_of_accuracies)   
#     sd_acc = np.std(list_of_accuracies)
#     average_mse = np.mean(list_of_accuracies)   
#     sd_mse = np.std(list_of_accuracies)

#     print(f'Accuracy average: {average_acc}, standard dev: {sd_acc}')
#     print(f'MSE average: {average_mse}, MSE standard dev: {sd_mse}')

#      # Lists for plotting
#     ids = list(accuracies.keys())
#     accuracy_scores = list(accuracies.values())

#    # Plotting
#     plt.figure(figsize=(10, 6))  
#     plt.plot(ids, accuracy_scores, marker='o', linestyle='-', color='b') 
#     plt.xlabel('ScrSubjectID')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy by ScrSubjectID')
#     plt.ylim([0, 1])  # Assuming accuracy is between 0 and 1
#     plt.grid(True)  # Adds a grid for easier readability
#     plt.show()

#     ######## TRIAL FIVE: Using limited dataset, person-specific random forest########
#     print('TRIAL FIVE: Person-specific random forest (hrv and hr only) ')
#     rf_data = complete_whoop_dataset
#     rf2_data = rf_data[['ScrSubjectID','Pre STAI State Anxiety', 'RHR', 'HRV']]


#     persons_data = rf2_data['ScrSubjectID'].unique()

#     accuracies = {}
#     list_of_accuracies = []
#     list_of_MSEs = []
#     for person in persons_data:
#         person_data = rf2_data[rf_data['ScrSubjectID'] == person]
#         if len(person_data)<10:
#              continue
#         X = person_data.drop('Pre STAI State Anxiety', axis=1)  
#         y = person_data['Pre STAI State Anxiety']
    
#         # Model #1: Ridge Regression
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
    
#         # Hyperparameters according to the paper
#         rf = RandomForestClassifier(n_estimators=1000, max_depth=2, max_features='sqrt', random_state=42, verbose=1)
        
#         # 10-fold CV
#         # n_splits = len(person_data) if len(person_data) < 10 else 10
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)
#         cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=kf)
        
#         # Fit the model
#         rf.fit(X_train_scaled, y_train)
#         feature_importances = rf.feature_importances_
        
#         # Evaluate the model
#         y_pred = rf.predict(X_test_scaled)

#         accuracy = accuracy_score(y_test, y_pred)
#         MSE = mean_squared_error(y_test, y_pred)


#         # Save accuracy
#         accuracies[person] = accuracy
#         list_of_accuracies.append(accuracy)
#         list_of_MSEs.append(MSE)

#     # for person, accuracy in accuracies.items():
#     #         print(f"Person {person}: accuracy = {accuracy}")

#     average_acc = np.mean(list_of_accuracies)   
#     sd_acc = np.std(list_of_accuracies)
#     average_mse = np.mean(list_of_accuracies)   
#     sd_mse = np.std(list_of_accuracies)

#     print(f'Accuracy average: {average_acc}, standard dev: {sd_acc}')
#     print(f'MSE average: {average_mse}, MSE standard dev: {sd_mse}')


#     # Lists for plotting
#     ids = list(accuracies.keys())
#     accuracy_scores = list(accuracies.values())

#    # Plotting
#     plt.figure(figsize=(10, 6))  
#     plt.plot(ids, accuracy_scores, marker='o', linestyle='-', color='b') 
#     plt.xlabel('ScrSubjectID')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy by ScrSubjectID')
#     plt.ylim([0, 1])  # Assuming accuracy is between 0 and 1
#     plt.grid(True)  # Adds a grid for easier readability
#     plt.show()
    
     ######## TRIAL SIX: Using full dataset, xgBoost########
    print('TRIAL SIX: Full dataset XGboost ')
    xg_data = complete_whoop_dataset
    xg1_data = xg_data.drop(columns=['Post STAI State Anxiety'])

    X = xg1_data.drop('Pre STAI State Anxiety', axis=1)  
    y = xg1_data['Pre STAI State Anxiety']

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
'''
    Begin person-specific trials
    Hybrid model
    -- over time? -- 
    '''
