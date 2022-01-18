# import the needed libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    This method recive database_file path
    and read the information from that database
    and store it into a dataframe 
    and return that dataframe
    '''
    # run the database file
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # run the messages table
    df = pd.read_sql_table('Messages', engine)
    #define X, y 
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    return X, y


def tokenize(text):
    '''
    This method receive a text and split it into words and store them into an array     
    and return the array     
    '''
    # Splitting text into words by strings of 
    # alphanumeric characters (a-z, A-Z, 0-9 and ‘_’)
    words=re.split(r'\W+', text)
    
    # Tokenize words from text
    tokens= word_tokenize(text)
    
    # Convert words to lower case
    words = [word.lower() for word in words]
    
    # Filter out stop words 
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words    


def build_model():
    '''
    This method builds the model 
    and return it
    '''
    # Build a pipeline
    pipeline =Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Parameters identifing 
    parameters = {
            'clf__estimator__warm_start': (True, False),
            'tfidf__smooth_idf': (True, False)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test):
    '''
    This method evaluate the model 
    '''
    y_prediction = model.predict(X_test)
    
    # index start from 0
    index=0

    # for-loop that rounds all the columns in Y_tset 
    for column in y_test:
        #Print column name
        print(column) 
        #Printing classification report for every class
        print(classification_report(y_test[column], y_prediction[:,index]))
        index=index+1

    # Calculating the accuracy for every calss    
    accuracy_class = (y_test == y_prediction).mean()
    print(accuracy_class)

    #Calculating the accuracy for the model    
    accuracy_model = (y_prediction == y_test.values).mean()
    print("The model accuracy is: \n",accuracy_model)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
