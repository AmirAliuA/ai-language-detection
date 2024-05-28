"""
    This code is a language detection model that uses the Naive Bayes algorithm 
    to classify a given text into one of the 22 languages present in the dataset. 
 
    The code imports various libraries such as pandas for data manipulation and analysis, 
    numpy for scientific computing and working with arrays, CountVectorizer for extracting features from text data, 
    train_test_split for splitting data into training and testing sets, and tabulate for printing out data in a formatted table.
 
    The datasets used in this code contain 39 languages (combined) with more than 1000 sentences from each language, and the output 
    should show the count of each language in the dataset. 
 
    The code then splits the data into training and test sets and trains the Naive Bayes algorithm on the training 
    set to predict the language of a given text. Finally, the code prints a table with the predicted language of the input text.
"""

import pandas as pd # used for data manipulation and analysis
import numpy as np # used for scientific computing and working with arrays
import os # used only for the filesize feature
from sklearn.feature_extraction.text import CountVectorizer # used for extracting features from text data
from sklearn.model_selection import train_test_split # used for splitting data into training and testing sets
from sklearn.naive_bayes import MultinomialNB # used for implementing Naive Bayes algorithm for classification
from tabulate import tabulate # used for printing out data in formatted table

def read_data(file_path):
    # reads a CSV file and returns a pandas DataFrame
    return pd.read_csv(file_path)

def preview_data(data):
    # prints the first rows of a pandas DataFrame
    print(tabulate(data.head(), headers='keys', tablefmt='psql', maxcolwidths=[30, 20]))

def get_null_values(data):
    # returns the sum of null values in each column of a pandas DataFrame
    return pd.DataFrame(data.isnull().sum(), columns=["Null Values"])

def print_null_values(data):
    # prints the sum of null values in each column of a pandas DataFrame
    null_values = get_null_values(data)
    print(tabulate(null_values, headers="keys", tablefmt="psql"))

def get_language_count(data, language_column):
    # returns the count of languages in a pandas DataFrame
    return pd.DataFrame(data[language_column].value_counts())

def print_language_count(data, language_column):
    # prints the count of languages in a pandas DataFrame
    language_count = get_language_count(data, language_column)
    print(tabulate(language_count, headers=["Language", "Count"], tablefmt="psql"))

def split_data(data, text_column, language_column, test_size = 0.33, random_state = 42):
    # splits the data into training and testing sets
    x = np.array(data[text_column])
    y = np.array(data[language_column])

    cv = CountVectorizer()
    X = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return cv, X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    # trains the model using Multinomial Naive Bayes algorithm
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model

def test_model(model, X_test, y_test):
    # tests the models accuracy
    return model.score(X_test, y_test)

def predict_language(model, cv, user_input):
    # predicts the language from the users input
    data = cv.transform([user_input]).toarray()
    output = model.predict(data)
    prediction_table = pd.DataFrame(output, columns=["Language"])
    
    return prediction_table

dataset1 = read_data("dataset1.csv")
dataset2 = read_data("dataset2.csv")

fileSize1 = os.path.getsize("dataset1.csv")
print("Dataset 1 file size: " + str(fileSize1))
fileSize2 = os.path.getsize("dataset2.csv")
print("Dataset 2 file size: " + str(fileSize2))

print("Data Preview for Dataset 1:\n")
preview_data(dataset1)

print("Data Preview for Dataset 2:\n")
preview_data(dataset2)

print("Dataset 1 Null Values:")
print_null_values(dataset1)

print("Dataset 2 Null Values:")
print_null_values(dataset2)

print("Dataset 1 Language Count:")
print_language_count(dataset1, "language")

print("Dataset 2 Language Count:")
print_language_count(dataset2, "Language")

cv1, X1_train, X1_test, y1_train, y1_test = split_data(dataset1, "Text", "language")
cv2, X2_train, X2_test, y2_train, y2_test = split_data(dataset2, "Text", "Language")

model1 = train_model(X1_train, y1_train)
accuracy1 = test_model(model1, X1_test, y1_test)

model2 = train_model(X2_train, y2_train)
accuracy2 = test_model(model2, X2_test, y2_test)

print("Accuracy of Model 1: {:.2f}%".format(accuracy1 * 100))
print("Accuracy of Model 2: {:.2f}%".format(accuracy2 * 100))

# print prediction table
user = input("\nEnter the text you want to detect: ")

data1 = cv1.transform([user]).toarray()
output1 = model1.predict(data1)
prediction_table1 = pd.DataFrame(output1, columns=["Language"])
print("\nPrediction for Dataset 1:\n", tabulate(prediction_table1, headers="keys", tablefmt="psql"))

data2 = cv2.transform([user]).toarray()
output2 = model2.predict(data2)
prediction_table2 = pd.DataFrame(output2, columns=["Language"])
print("\nPrediction for Dataset 2:\n", tabulate(prediction_table2, headers="keys", tablefmt="psql"))