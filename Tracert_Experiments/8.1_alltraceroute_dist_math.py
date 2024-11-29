import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import multiprocessing
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Define model names as strings
naive_model_name = 'naive'
simple_model_name = 'simple'
double_model_name = 'double'

def load_data(iteration):
    # Get the absolute path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the train and test pickles for the current iteration
    train_pickle_path = os.path.join(script_dir, 'train_test_pickles', f'train_df_{iteration}.pickle')
    test_pickle_path = os.path.join(script_dir, 'train_test_pickles', f'test_df_{iteration}.pickle')

    # Load the pickles
    train_df = pd.read_pickle(train_pickle_path)
    test_df = pd.read_pickle(test_pickle_path)

    # Define X_train, y_train, X_test, and y_test based on your data
    X_train = train_df['normalizzed_distance'].values.reshape(-1,1)
    y_train = train_df['normalizzed_rtt'].values
    
    X_test = test_df['normalizzed_distance'].values.reshape(-1,1)
    y_test = test_df['normalizzed_rtt'].values
    
    #get the training std 
    trainrtt_mean = train_df['last_rtt'].mean()
    trainrtt_std = train_df['last_rtt'].std()
    
    return trainrtt_std, test_df

def naive_model(iteration, results_naive, trainrtt_std,test_df):
    # Using the naive forecast
    test_df = test_df.assign(naive=test_df['normalizzed_rtt'].shift(1)) # next sample is the previous one

    # Replace NaN at top of value column with 0
    test_df['naive'] = test_df['naive'].fillna(method='ffill').fillna(0)

    # Testing the prediction accuracy for naive forecast
    rmse_naive = np.sqrt(mean_squared_error(test_df['normalizzed_rtt'], test_df['naive']))
    drmse_naive = rmse_naive * trainrtt_std
    
    # Append the results to the list
    results_naive.append({'Iteration': iteration, 'RMSE': rmse_naive, 'Denormalized_RMSE': drmse_naive})
    
    print(f'{naive_model_name} evaluation results saved')
    
    
def simple_model(iteration, results_simple, trainrtt_std,test_df):
    from statsmodels.tsa.api import SimpleExpSmoothing

    fit1 = SimpleExpSmoothing(test_df['normalizzed_rtt']).fit()
    test_df['Simple-smoothing'] = SimpleExpSmoothing(test_df['normalizzed_rtt']).fit().fittedvalues

    rmse_esm = np.sqrt(mean_squared_error(test_df['normalizzed_rtt'], test_df['Simple-smoothing']))
    drmse_esm = rmse_esm * trainrtt_std
    
    # Append the results to the list
    results_simple.append({'Iteration': iteration, 'RMSE': rmse_esm, 'Denormalized_RMSE': drmse_esm})
    
    print(f'{simple_model_name} evaluation results saved')
    
    
def double_model(iteration, results_double, trainrtt_std,test_df):
    from statsmodels.tsa.api import Holt

    # Specify your own values for alpha and beta
    alpha_value = 0.6
    beta_value = 0.2

    # Fitting (adjusting) the data to Holt's linear method with specified alpha and beta
    fit2 = Holt(test_df['normalizzed_rtt']).fit()
    test_df['Double-smoothing'] = Holt(test_df['normalizzed_rtt']).fit(smoothing_level=alpha_value, smoothing_slope=beta_value).fittedvalues

    # Testing the prediction accuracy for Holt's linear method
    se = (test_df['normalizzed_rtt'] - test_df['Double-smoothing']) ** 2
    mse_desm = se.mean()

    # Print the root Mean Squared Error for Holt's linear method
    rmse_desm = np.sqrt(mean_squared_error(test_df['normalizzed_rtt'], test_df['Double-smoothing']))
    drmse_desm = rmse_desm * trainrtt_std
    
    # Append the results to the list
    results_double.append({'Iteration': iteration, 'RMSE': rmse_desm, 'Denormalized_RMSE': drmse_desm})
    
    print(f'{double_model_name} evaluation results saved')
    
from multiprocessing import Manager
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    with Manager() as manager:
        # Create shared lists for each model's results
        results_naive = manager.list()
        results_simple = manager.list()
        results_double = manager.list()
        
        for iteration in range(20):
            # Load the data for the current iteration
            trainrtt_std,test_df = load_data(iteration)
            
            # Create processes for each model
            processes = []
            
            processes.append(multiprocessing.Process(target=naive_model, args=(iteration, results_naive,trainrtt_std,test_df)))
            processes.append(multiprocessing.Process(target=simple_model, args=(iteration, results_simple,trainrtt_std,test_df)))
            processes.append(multiprocessing.Process(target=double_model, args=(iteration, results_double,trainrtt_std,test_df)))
            
            
            #Start the processes
            for process in processes:
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()
                
        # Convert the shared lists to a list of dictionaries
        list_of_dicts_naive = list(results_naive)
        list_of_dicts_simple = list(results_simple)
        liist_of_dicts_double = list(results_double)
        
        # Convert the lists of dictionaries to dataframes
        results_df_naive = pd.DataFrame(list_of_dicts_naive)
        results_df_simple = pd.DataFrame(list_of_dicts_simple)
        results_df_double = pd.DataFrame(liist_of_dicts_double)
        
        # Define the absolute path for the results folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_folder = os.path.join(script_dir, 'model_dist_results')

        # Check if the folder exists, create it if not
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            
        results_df_naive.to_csv(os.path.join(results_folder, f'evaluation_results_naive.csv'), index=False)
        results_df_simple.to_csv(os.path.join(results_folder, f'evaluation_results_simple.csv'), index=False)   
        results_df_double.to_csv(os.path.join(results_folder, f'evaluation_results_double.csv'), index=False)
    