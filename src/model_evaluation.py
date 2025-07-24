#This file deals with Model Evaluation.

import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live

#Logs directory
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

#Logging Configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

#Load Param File
def load_params(params_path: str) -> dict:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML Error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error: %s',e)
        raise

#Load Model
def load_model(file_path: str):
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s',file_path)
        return model
    except FileNotFoundError:
        logger.debug('Model not found at %s',file_path)
        raise
    except Exception as e:
        logger.debug('Unexpected error %s',e)
        raise

#Load Test Data
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug('Test Data Loaded from %s',file_path)
        return df
    except FileNotFoundError:
        logger.error('File not found: %s',file_path)
        raise
    except pd.error.ParseError as e:
        logger.error('Error reading file: %s',e)
        raise
    except Exception as e:
        logger.debug('Unexpected error %s',e)
        raise

#Evaluation of the model
def evaluate_model(clf,X_test: np.ndarray,y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)
        
        metrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        
        logger.debug('Model Evaluation are accuracy: %f , precision : %f, recall : %f, auc: %f', accuracy, precision, recall, auc)
        return metrics_dict
    except Exception as e:
        logger.error('Unexpected error %s',e)
        raise

#Save metrics
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics save to %s', file_path)
    except Eception as e:
        logger.error('Unexpected error occured %s',e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        clf = load_model('./model/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        
        metrics = evaluate_model(clf, X_test, y_test)
            
            
            
        # Experiment Tracking Uisng DVClive       
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test,y_test))
            live.log_metric('precision', precision_score(y_test,y_test))
            live.log_metric('recall',recall_score(y_test,y_test))
            live.log_params(params)
              
        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error('Unexpected error : %s',e)
        print('Error{e}')
        
if __name__ == "__main__":
    main()