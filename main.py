# Import Libraries
import pandas as pd
import numpy as np
#importmatplotlib.pyplot as plt
#importseabornassns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import requests
import pandas as pd
import numpy as np
import pickle 
local_model1 = pickle.load(open('Model1.pickle', 'rb'))
local_model2 = pickle.load(open('Model2.pickle', 'rb'))
local_model3 = pickle.load(open('Model3.pickle', 'rb'))
local_model4 = pickle.load(open('Model4.pickle', 'rb'))
local_model5 = pickle.load(open('Model5.pickle', 'rb'))
url='http://localhost:8501/'
#importplotly.express 
#as px
st.set_option('deprecation.showPyplotGlobalUse', False)
#Import EDA
from eda import eda
# Load data
dataset = load_iris()
# Create dataframe with iris data
print(dataset)
data = dataset.data
target_names = dataset.target_names # Classes
feature_names = dataset.feature_names # Columns
target = dataset.target # Output
df = pd.DataFrame(data, columns = feature_names)
# Make target a series
target = pd.Series(target)
# Streamlit
# Set up App

st.set_page_config(page_title=" ML Dashboard", 
                   layout="centered",
                   initial_sidebar_state="auto")
# Add title and markdown decription
st.title("EDA and Predictive Modelling Dashboard")

# define sidebar and sidebar options
options = ["EDA", "Predictive Modelling"]
selected_option = st.sidebar.selectbox("Select an option", options)
# Do EDA
if selected_option == "EDA":
    # Call/invoke EDA function from ead.py
    eda(df, target_names, feature_names, target)    
    #eda(df, target_names, feature_names, target)         
# Predictive Modelling
elif selected_option == "Predictive Modelling":
    st.subheader("Predictive Modelling")
    st.write("Choose a Model with a transform type if needed from the options below:")
    X = df.values
    Y = target.values
    
    test_proportion = 0.30
    seed = 5
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_proportion, random_state=seed)
    transform_options = ["None", 
                         "StandardScaler", 
                         "Normalizer", 
                         "MinMaxScaler"]
    transform = st.selectbox("Select data transform",
                             transform_options)
    if transform == "StandardScaler":
        input_transform="StandardScaler"
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "Normalizer":
        input_transform="Normalizer"
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif transform == "MinMaxScaler":
        input_transform="MinMaxScaler"
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train
        X_test = X_test
    classifier_list = ["LogisticRegression",
                       "SVM",
                       "DecisionTree",
                       "KNeighbors",
                       "RandomForest"]
    classifier = st.selectbox("Select classifier", classifier_list)
    
    


# Extract model and data transform type
model = df['model'][0]
data = df['data'][0]
st.write(model, data)
# Create json object
request_data = json.dumps({'model': any(model), 'data': any(data)})
# Send to the endpoint (RESTful API)
requestpost = requests.post(url, request_data)
# Retrieve data from RESTful API
res = requestpost.json()
st.write(res)