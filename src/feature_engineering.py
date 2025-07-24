# File to create features

#Import
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

#Logging directory
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True) 

#Log Configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,"feature_engineering.log")
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
        logger.debug('Parameters from %s loaded',params_path)
        return params
    except FileNotFoundError:
        logger.error('File Not Found at %s',params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error %s',e)
        raise

#Load Data File
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParseError as e:
        logger.error('Failed to parse CSV: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s',e)
        raise
    
#Create TFIDF features
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test
        
        logger.debug('TFIDF Vectors created for Train and Test Data')
        return train_df, test_df
    except Exception as e:
        logger.error('Error during TFIDF transformation: %s',e)
        raise

#Saving Transformed Data
def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('Data Saved:%s',file_path)
    except Exception as e:
        logger.error('Unexpected error: %s',e)
        raise
        
#Main Function
def main():
    try:
        params = load_params(params_path='params.yaml')
        max_features = params['feature_engineering']['max_features']
        
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        save_data(train_df, os.path.join('./data','processed','train_tfidf.csv'))
        save_data(test_df, os.path.join('./data','processed','test_tfidf.csv'))
    except Exception as e:
        logger.error('Failed Features Generation: %s',e)
        print(f"Error:{e}")

if __name__ == "__main__":
    main()
