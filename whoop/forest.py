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
from util import create_whoop_data
from util import run_rf, run_xgb
from figs import rf_plots

if __name__ == '__main__':
    data = pd.read_excel('BWPilot_CombinedData_20210803_fordryad_addvars_cleaned_noround2_v3.xlsx', engine='openpyxl')
    whoop_data = create_whoop_data(data)
    
    ######## PART ONE: Using full dataset ########
    ## TRIAL ONE full ##
    print('TRIAL ONE Full Dataset')
    trial_1_acc, trial_1_precision, trial_1_recall, trial_1_f1 = run_rf(whoop_data, full_metrics=True)
    ## do some stuff with these
    
    ## TRIAL TWO, RHR, HRV, Sleep Score, Hours of Sleep, Sleep Efficiency, Respiration Rate ##
    print('TRIAL TWO smaller dataset')
    small_trial_data = whoop_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'Sleep Score', 'Hours of Sleep', 'Sleep Efficiency', 'Respiration Rate', 'ScrSubjectID']]
    trial_2_acc, trial_2_precision, trial_2_recall, trial_2_f1 = run_rf(small_trial_data, full_metrics=True)
    
    ## TRIAL THREE, RHR, HRV ##
    print('TRIAL THREE HRV RHR only')
    limited_trial_data = whoop_data[['Pre STAI State Anxiety', 'RHR', 'HRV', 'ScrSubjectID']]
    trial_3_acc, trial_3_precision, trial_3_recall, trial_3_f1 = run_rf(limited_trial_data, full_metrics=True)
    
    ######## PART TWO: Person-specific models ########
    ## TRIAL FOUR, Full dataset ##
    print('TRIAL FOUR: Person-specific random forest (full dataset)')
    persons_data = whoop_data['ScrSubjectID'].unique()

    accuracies = {}
    list_of_accuracies = []
    for person in persons_data:
        person_data = whoop_data[whoop_data['ScrSubjectID'] == person]
        if len(person_data) < 10:
            continue
        indiv_acc, _, _, _ = run_rf(person_data)
        if indiv_acc != -1:
            accuracies[person] = indiv_acc
            list_of_accuracies.append(indiv_acc)

    average_acc = np.mean(list_of_accuracies)   
    sd_acc = np.std(list_of_accuracies)

    print(f'Accuracy average: {average_acc}, standard dev: {sd_acc}')
    ids = list(accuracies.keys())
    accuracy_scores = list(accuracies.values())
    rf_plots(ids, accuracy_scores)

    ## TRIAL FIVE: HRV, RHR only 
    print('TRIAL FIVE: Person-specific random forest (hrv and hr only) ')
    rf2_data = whoop_data[['ScrSubjectID','Pre STAI State Anxiety', 'RHR', 'HRV']]
    persons_data = rf2_data['ScrSubjectID'].unique()

    accuracies = {}
    list_of_accuracies = []
    
    for person in persons_data:
        person_data = rf2_data[rf2_data['ScrSubjectID'] == person]
        if len(person_data) < 10:
             continue
        indiv_acc, _, _, _ = run_rf(person_data)
        if indiv_acc != -1:
            accuracies[person] = indiv_acc
            list_of_accuracies.append(indiv_acc)

    average_acc = np.mean(list_of_accuracies)   
    sd_acc = np.std(list_of_accuracies)

    print(f'Accuracy average: {average_acc}, standard dev: {sd_acc}')

    ids = list(accuracies.keys())
    accuracy_scores = list(accuracies.values())
    rf_plots(ids, accuracy_scores)
    
    ######## PART THREE: XGB ########
    ## TRIAL SIX: Full dataset XGB ##
    print('TRIAL SIX: Full dataset XGboost ')
    xg1_data = whoop_data.drop(columns=['Post STAI State Anxiety', 'ScrSubjectID'])
    
    trial_6_acc = run_xgb(xg1_data)
    
    ## TRIAL SEVEN: HRV and RHR only
    xg2_data = whoop_data[['Pre STAI State Anxiety', 'HRV', 'RHR']]
    trial_7_acc = run_xgb(xg2_data)

