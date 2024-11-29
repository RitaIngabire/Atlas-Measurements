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

# Create a folder named  to store the evaluation results
results_folder = 'model_dist_results'
os.makedirs(results_folder, exist_ok=True)

# Define model names as strings
linear_model_name = 'linear'
rf_model_name = 'rf'
dt_model_name = 'dt'
xgb_model_name = 'xgb'
knn_model_name = 'knn'
svr_model_name = 'svr'

def linear_model(iteration, results_linear,X_train, y_train, X_test, y_test,trainrtt_std):
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse_lm = np.sqrt(mean_squared_error(y_test, y_pred))
    drmse_lm = rmse_lm * trainrtt_std
    
    # Append the results to the list
    results_linear.append({'Iteration': iteration, 'RMSE': rmse_lm, 'Denormalized_RMSE': drmse_lm})

    #print(f'{linear_model_name} evaluation results saved')
    print(results_linear)
    

def rf_model(iteration,results_rf, X_train, y_train, X_test, y_test,trainrtt_std):
    # Train a random forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
    drmse_rf = rmse_rf * trainrtt_std
    
    # Append the results to the list
    results_rf.append({'Iteration': iteration, 'RMSE': rmse_rf, 'Denormalized_RMSE': drmse_rf})

    print(f'{rf_model_name} evaluation results saved ')
    

def dt_model(iteration, results_dt,X_train, y_train, X_test, y_test,trainrtt_std):
    # Train a dt model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred))
    drmse_dt = rmse_dt * trainrtt_std
    
    # Append the results to the list
    results_dt.append({'Iteration': iteration, 'RMSE': rmse_dt, 'Denormalized_RMSE': drmse_dt})

    print(f'{dt_model_name} evaluation results saved to')
    
def xbg_model(iteration, results_xgb,X_train, y_train, X_test, y_test,trainrtt_std):
        
        regressor= XGBRegressor()
        
        param_grid = {"max_depth":    [4, 5, 6],
                "n_estimators": [50, 60, 70],
                "learning_rate": [0.01, 0.015]}
        
        grid_search = GridSearchCV(regressor, param_grid).fit(X_train, y_train)
        
        regressor=XGBRegressor(learning_rate = grid_search.best_params_["learning_rate"],
                        n_estimators  = grid_search.best_params_["n_estimators"],
                        max_depth     = grid_search.best_params_["max_depth"],
                        objective     = 'reg:squarederror')

        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        
        # Calculate evaluation metrics
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
        drmse_xgb = rmse_xgb * trainrtt_std
        
        # Append the results to the list
        results_xgb.append({'Iteration': iteration, 'RMSE': rmse_xgb, 'Denormalized_RMSE': drmse_xgb})

        print(f'{xgb_model_name} evaluation results saved to')
        
    
def knn_model(iteration, results_knn, X_train, y_train, X_test, y_test,trainrtt_std):
    # Create an instance of the KNN regressor
    knn = KNeighborsRegressor(n_neighbors=5,weights='distance')
    
    # Fit the model on the training data
    knn.fit(X_train, y_train)

    # Make a prediction on the testing data
    y_pred = knn.predict(X_test)
    
    knn_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    knn_drmse = knn_rmse * trainrtt_std
    
    # Append the results to the list
    results_knn.append({'Iteration': iteration, 'RMSE': knn_rmse, 'Denormalized_RMSE': knn_drmse })

    print(f'{knn_model_name} evaluation results saved to')
    

def svr_model(iteration, results_svr, X_train, y_train, X_test, y_test,trainrtt_std):
    # Create a linear SVR
    regressor = SVR(kernel='linear')

    # Train a  regressor
    regressor.fit(X_train[:1000], y_train[:1000])
    
    # Make predictions on the test set using the trained regressor
    y_pred = regressor.predict(X_test)

    # Calculate the loss function
    svr_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    svr_drmse = svr_rmse * trainrtt_std
    
    # Append the results to the list
    results_svr.append({'Iteration': iteration, 'RMSE': svr_rmse, 'Denormalized_RMSE': svr_drmse})

    print(f'{svr_model_name} evaluation results saved to')


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
    
    return X_train, y_train, X_test, y_test, trainrtt_std


from multiprocessing import Manager
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    with Manager() as manager:
        # Create shared lists for each model's results
        results_linear = manager.list()
        results_rf = manager.list()
        results_dt = manager.list()
        results_xgb = manager.list()
        results_knn = manager.list()
        results_svr = manager.list()

        for iteration in range(20):
            # Load the data for the current iteration
            X_train, y_train, X_test, y_test, trainrtt_std = load_data(iteration)
            
            # Create processes for each model
            processes = []
            
            processes.append(multiprocessing.Process(target=linear_model, args=(iteration, results_linear, X_train, y_train, X_test, y_test, trainrtt_std)))
            processes.append(multiprocessing.Process(target=rf_model, args=(iteration, results_rf, X_train, y_train, X_test, y_test, trainrtt_std)))
            processes.append(multiprocessing.Process(target=dt_model, args=(iteration, results_dt, X_train, y_train, X_test, y_test, trainrtt_std)))
            processes.append(multiprocessing.Process(target=xbg_model, args=(iteration, results_xgb, X_train, y_train, X_test, y_test, trainrtt_std)))
            processes.append(multiprocessing.Process(target=knn_model, args=(iteration, results_knn, X_train, y_train, X_test, y_test, trainrtt_std)))
            processes.append(multiprocessing.Process(target=svr_model, args=(iteration, results_svr, X_train, y_train, X_test, y_test, trainrtt_std)))
            
            # Start the processes
            for process in processes:
                process.start()

            # Wait for all processes to finish
            for process in processes:
                process.join()
                
        # Convert the shared lists to a list of dictionaries
        list_of_dicts_linear = list(results_linear)
        list_of_dicts_rf = list(results_rf)
        list_of_dicts_dt = list(results_dt)
        list_of_dicts_xgb = list(results_xgb)
        list_of_dicts_knn = list(results_knn)
        list_of_dicts_svr = list(results_svr)

        # Convert the lists of dictionaries to dataframes
        results_df_linear = pd.DataFrame(list_of_dicts_linear)
        results_df_rf = pd.DataFrame(list_of_dicts_rf)
        results_df_dt = pd.DataFrame(list_of_dicts_dt)
        results_df_xgb = pd.DataFrame(list_of_dicts_xgb)
        results_df_knn = pd.DataFrame(list_of_dicts_knn)
        results_df_svr = pd.DataFrame(list_of_dicts_svr)

        print(results_df_linear)
        # Define the absolute path for the results folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_folder = os.path.join(script_dir, 'model_dist_results')

        # Check if the folder exists, create it if not
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Save dataframes to CSV files
        
        results_df_linear.to_csv(os.path.join(results_folder, f'evaluation_results_linear.csv'), index=False)
        results_df_rf.to_csv(os.path.join(results_folder, f'evaluation_results_rf.csv'), index=False)
        results_df_dt.to_csv(os.path.join(results_folder, f'evaluation_results_dt.csv'), index=False)
        results_df_xgb.to_csv(os.path.join(results_folder, f'evaluation_results_xgb.csv'), index=False)
        results_df_knn.to_csv(os.path.join(results_folder, f'evaluation_results_knn.csv'), index=False)
        results_df_svr.to_csv(os.path.join(results_folder, f'evaluation_results_svr.csv'), index=False)
