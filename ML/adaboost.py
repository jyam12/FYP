import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
import joblib

class adaboost:
    # Parse and preprocess relevant columns for all levels
    def __init__(self,X_train,y_train,X_eval):

        self.X_train = X_train.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.X_eval = X_eval.copy(deep=True)

    # Train AdaBoost Regressor
    def train(self):
        adaboost_model = AdaBoostRegressor(n_estimators=50, random_state=42)
        adaboost_model.fit(self.X_train, self.y_train)
        # Save the models
        joblib.dump(adaboost_model, "ML/adaboost_model.pkl") 

    def predict(self):
        adaboost_model = joblib.load("ML/adaboost_model.pkl")
        return adaboost_model.predict(self.X_eval)