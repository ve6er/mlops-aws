## Create Model for Spam Identification

import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

#Logs directory

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#Log configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#Load Parameter File
def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',file)
        return params
    except FileNotFoundError:
        logger.error('File Not Found at %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error:%s',e)
        raise

#Load Data
def load_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s',file_path)
        return df
    except Exception as e:
        logger.error('Unexpected Error %s',e)
        raise
    except FileNotFoundError:
        logger.debug('File not found at %s',file_path)
        raise
    except pf.errors.ParseError as e:
        logger.debug('Cannot parse file: %s',e)
        raise

#Train model
def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    try:
        if X_train.shape[0]!= y_train.shape[0]:
            raise ValueError('Sample mismatch from X_train and y_train')
        else:
            logger.debug('Intializing RF Model params from %s',params)
            clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])
            
            logger.debug('Model training starred with %d samples',X_train.shape[0])
            clf.fit(X_train,y_train)
            logger.debug('Training Completed')
            
            return clf
    except ValueError as e:
        logger.error('Value Error during Model training %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error: %s',e)
        raise
    
#Save Model
def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved to %s',file_path)
    except FileNotFoundError:
        logger.debug('File Not Found at %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected Error %s',e)
        raise

#Main
def main():
    try:
        params = load_params('params.yaml')['model_building']
        train_data = load_csv('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        clf = train_model(X_train, y_train, params)
        
        model_save_path = 'model/model.pkl'
        save_model(clf, model_save_path)
    
    except Exception as e:
        logger.error('Failed to build model %s',e)
        print(f'Error{e}')
        

if __name__ == "__main__":
    main()