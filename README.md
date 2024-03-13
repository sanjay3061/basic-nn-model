# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

A neural network is a computer program inspired by how our brains work. It's used to solve problems by finding patterns in data. Imagine a network of interconnected virtual "neurons." Each neuron takes in information, processes it, and passes it along. A Neural Network Regression Model is a type of machine learning algorithm that is designed to predict continuous numeric values based on input data. It utilizes layers of interconnected nodes, or neurons, to learn complex patterns in the data. The architecture typically consists of an input layer, one or more hidden layers with activation functions, and an output layer that produces the regression predictions. This model can capture intricate relationships within data, making it suitable for tasks such as predicting prices, quantities, or any other continuous numerical outputs.

## Neural Network Model

![image](https://github.com/krithygowthamn/basic-nn-model/assets/122247810/c4ca3e0d-fffd-4834-8dfd-4051489d82a3)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Gowtham
### Register Number:212222220013
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Gowtham').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'input':'int'})
dataset1 = dataset1.astype({'output':'int'})
dataset1.head()
X = dataset1[['input']].values
y = dataset1[['output']].values
X
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
MinMaxScaler()
X_train1 = Scaler.transform(X_train)
ai_brain=Sequential([Dense(units=10,input_shape=[1]),Dense(units=5),Dense(units=1,activation="relu")])
ai_brain.compile(optimizer="rmsprop",loss="mae")
ai_brain.fit(X_train1,y_train,epochs=2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[18]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information
![image](https://github.com/krithygowthamn/basic-nn-model/assets/122247810/cd16b328-9e91-43d6-b3f2-e3a8824baf41)

## OUTPUT
### Training Loss Vs Iteration Plot
![image](https://github.com/krithygowthamn/basic-nn-model/assets/122247810/363a3f16-3265-495b-b6f9-15c59337c75f)

### Test Data Root Mean Squared Error
![image](https://github.com/krithygowthamn/basic-nn-model/assets/122247810/c6c95b2c-034f-4d5d-b0c5-ac986e381050)

## New Sample Data Prediction
![image](https://github.com/krithygowthamn/basic-nn-model/assets/122247810/ec24ceda-b098-4045-ba4b-b3d8cc0730d7)

## RESULT
Thus to develop a neural network regression model for the dataset created is successfully executed.
