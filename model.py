import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# Load the csv file
df = pd.read_csv('Invistico_Airline.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Split the dataframe into features (X) and target variable (y)
X1 = df.drop('satisfaction', axis=1)
#x2 = df.iloc[:,[1]].values
#new
df = pd.get_dummies(df,columns=["Gender"],drop_first=True)


X = x = df.iloc[:, [12,7,14,13]].values
y = df['satisfaction']
#nwe
'''
# Encode categorical variables
le = LabelEncoder()

for col in X1.columns:
    if X1[col].dtype == 'object':
        X1[col] = le.fit_transform(X1[col])


    
# Scale numerical columns
scaler = StandardScaler()
X1[['Age', 'Flight Distance', 'Departure Delay in Minutes']] = scaler.fit_transform(X1[['Age', 'Flight Distance', 'Departure Delay in Minutes']])
'''
#X = x = df.iloc[:, [1,12,7,14,13]].values
#y = df['satisfaction']


categorical_columns = ['satisfaction', 'Gender', 'Customer Type', 'Type of Travel', 'Class', 'Seat comfort',
                       'Departure/Arrival time convenient', 'Food and drink', 'Gate location',
                       'Inflight wifi service', 'Inflight entertainment', 'Online support',
                       'Ease of Online booking', 'On-board service', 'Leg room service',
                       'Baggage handling', 'Checkin service', 'Cleanliness', 'Online boarding']
# Numerical columns
numerical_columns = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
'''
# Create an instance of OneHotEncoder
one_hot_encoder = OneHotEncoder()

# Fit and transform the categorical columns
X_encoded = one_hot_encoder.fit_transform(X[categorical_columns]).toarray()

# Concatenate the encoded features with the remaining numerical features
X_encoded = np.concatenate((X_encoded, X[numerical_columns]), axis=1)
'''
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Create an instance of RandomForestClassifier
model = RandomForestClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Get the predicted probabilities for class 1 (satisfaction = 'satisfied')
y_prob = model.predict_proba(X_test)[:, 1]

# Save the model as a pickle file
pickle.dump(model, open("model.pkl", "wb"))

