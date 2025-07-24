#File to process train and test data

#Imports
import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import nltk
nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
#Download nltk corpus
#nltk.download("all")

#Logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

#Setup console Logger
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

#Adding Logs to File
log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path) 
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text: str) -> str:
    """
        Transforms input text.
    """
    lemmatizer = WordNetLemmatizer()
    text = text.lower() #Convert text to lowercase
    text = nltk.word_tokenize(text) #Tokenize by words
    text = [word for word in text if word.isalnum()] #Select alphanumeric words only
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation] #Remove stopwords and punctuation
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text) #Convert list to string and return

def preprocess_df(df: pd.DataFrame, text_column: str, target_column: str) -> pd.DataFrame:
    """
        Preprocess DataFrame
    """
    try:
        logger.debug("Starting preprocessing for DataFrame")
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column]) #Encode Target Column
        logger.debug('Target Column Encoded.')
        
        df = df.drop_duplicates(keep='first') #Drop Duplicate Values
        logger.debug('Duplicate Removed')
        
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text Column Transformed.')
        return df
    except KeyError as e:
        logger.error('Column not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error during text preprocessing: %s',e)
        raise

def main(text_column='text', target_column='target'):
    try:
        #Load CSV Files
        train_data = pd.read_csv('./data/raw/train_data.csv')
        test_data = pd.read_csv('./data/raw/test_data.csv')
        logger.debug('Data Loaded Properly.')
        
        #Transform Data
        train_processed_data = preprocess_df(train_data, text_column,target_column)
        test_processed_data = preprocess_df(test_data,text_column,target_column)
        logger.debug('Data Transformed Correctly')
        
        #Store Transformed Data
        data_path = os.path.join("./data",'interim')
        os.makedirs(data_path,exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index = False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index = False)
        
        logger.debug('Processed Data is Saved: %s', data_path)
    except FileNotFoundError as e:
        logger.error('File Not Found: %s',e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s',e)
        raise
    except Exception as e:
        logger.error('Failed to complete preprocesing: %s',e)
        print(f"Error: %s",e)
    
if __name__ == "__main__":
    main() 
