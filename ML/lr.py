import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

class linear_regression:
    # Parse and preprocess relevant columns for all levels
    def __init__(self,X_train,y_train,X_eval):

        self.X_train = X_train.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.X_eval = X_eval.copy(deep=True)

    # Train Linear Regression model
    def train(self):
        lr_model = LinearRegression()
        lr_model.fit(self.X_train, self.y_train)
        # Save the models
        joblib.dump(lr_model, "ML/lr_model.pkl") 

    def predict(self):
        lr_model = joblib.load("ML/lr_model.pkl")
        return lr_model.predict(self.X_eval)