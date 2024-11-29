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
arima_model_name = 'arima'
prophet_model_name = 'prophet'

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
    
    return X_train, y_train, X_test, y_test, trainrtt_std,train_df,test_df



def arima_model(iteration, results_arima, X_train, y_train, X_test, y_test,trainrtt_std):
    #Importing the module
    import pmdarima as pm
    

    #first model 
    arima_model = pm.auto_arima(y_train,test="adf",trace=True,error_action='ignore',supress_warnings = True)
    
    # Get the best order
    order = arima_model.order
    model = sm.tsa.ARIMA(y_test, order=order)
    arima_model = model.fit()
    
    # Make predictions on the test set
    arima_predictions = arima_model.fittedvalues
    
    rmse_arima =  np.sqrt(mean_squared_error(arima_predictions,y_test))
    drmse_arima = rmse_arima * trainrtt_std
    
    # Append the results to the list
    results_arima.append({'Iteration': iteration, 'RMSE': rmse_arima , 'Denormalized_RMSE': drmse_arima })
    
    print(f'{arima_model_name} evaluation results saved')


def prophet_model(iteration, results_prophet, train_df, test_df,y_train, X_test, y_test, trainrtt_std):
    # Import the Prophet class from fbprophet
    from prophet import Prophet 
    
    #extract the test data from the test_df
    prophet_train = train_df.rename(columns={'date':'ds', 'normalizzed_rtt':'y'})
    prophet_train = prophet_train[['ds', 'y']]
    
    #extract the test data from the test_df
    prophet_test = test_df.rename(columns={'date':'ds', 'normalizzed_rtt':'y'})
    prophet_test = prophet_test[['ds', 'y']]
    y_test = prophet_test['y']

    # Create a new Prophet object
    model = Prophet()
    
    # Fit the model to the historical data
    model.fit(prophet_train)

    # Make In-Sample Predictions
    in_sample_forecast = model.predict(prophet_test) 
    
    # Calculate the RMSE
    rmse_prophet = np.sqrt(mean_squared_error(y_test, in_sample_forecast['yhat']))
    drmse_prophet = rmse_prophet * trainrtt_std
    
    # Append the results to the list
    results_prophet.append({'Iteration': iteration, 'RMSE': rmse_prophet, 'Denormalized_RMSE': drmse_prophet })

    print(f'{prophet_model_name} evaluation results saved')
    
    
from multiprocessing import Manager
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    with Manager() as manager:
        # Create shared lists for each model's results
        results_arima = manager.list()
        results_prophet = manager.list()
        
        for iteration in range(5):
            # Load the data for the current iteration
            X_train, y_train, X_test, y_test, trainrtt_std,train_df,test_df = load_data(iteration)
            
            # Create processes for each model
            processes = []
            
            processes.append(multiprocessing.Process(target=arima_model, args=(iteration, results_arima, X_train, y_train, X_test, y_test, trainrtt_std)))
            processes.append(multiprocessing.Process(target=prophet_model, args=(iteration, results_prophet, train_df, test_df,y_train, X_test, y_test, trainrtt_std)))
            
            #Start the processes
            for process in processes:
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()
                
        # Convert the shared lists to a list of dictionaries
        list_of_dicts_arima = list(results_arima)
        list_of_dicts_prophet = list(results_prophet)
        
        # Convert the lists of dictionaries to dataframes
        results_df_arima = pd.DataFrame(list_of_dicts_arima)
        results_df_prophet = pd.DataFrame(list_of_dicts_prophet)
        
        # Define the absolute path for the results folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_folder = os.path.join(script_dir, 'model_dist_results')

        # Check if the folder exists, create it if not
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            
        results_df_arima.to_csv(os.path.join(results_folder, f'evaluation_results_arima.csv'), index=False)
        results_df_prophet.to_csv(os.path.join(results_folder, f'evaluation_results_prophet.csv'), index=False)