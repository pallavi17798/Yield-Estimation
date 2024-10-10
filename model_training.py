from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow import keras
import tensorflow
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def train_model(df,model,epochs,crop_name):
    '''
    This function trains all the models for the given number of epoch
    '''
    X_tmp = df.iloc[:,:-7]
    y = df[crop_name]
    
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_tmp)
    
    
    #Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    history = model.fit(x=X_train, y=y_train, validation_split=0.2,batch_size=32, epochs=epochs)
    
    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    
    #Predict on the test data
    y_pred = model.predict(X_test)
    print(len(y_pred))
    
    # Plot the prediction vs actual values
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Prediction vs Actual')
    plt.show()
    
    y_pred = y_pred.reshape(len(y_test),)
    # Calculate the error
    print(f"y_test : {y_test.shape}\ny_pred : {y_pred.shape}")
    error = y_test - y_pred.reshape(len(y_test),)

    # Plot the distribution of error
    plt.hist(error, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Error')
    plt.show()
    
    # Calculate the mean squared error
    mse = np.mean((y_test - y_pred)**2)
    
    #Calculate the mean absolute error
    mae = np.mean(abs(y_test - y_pred))
    
    #Calculate root mean squared error
    rmse = np.sqrt(mse)
    
    #Calculate R2
    r2 = r2_score(y_test, y_pred)
    
    # Plot the evaluation metrics
    metrics = ['MAE', 'MSE', 'RMSE', 'R2']
    values = [mae, mse, rmse, r2]

    plt.bar(metrics, values)
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Evaluation Metrics')
    plt.show()
  