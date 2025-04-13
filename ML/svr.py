import pandas as pd
from sklearn.svm import SVR
import joblib

class svr:
    # Parse and preprocess relevant columns for all levels
    def __init__(self,X_train,y_train,X_eval):

        self.X_train = X_train.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.X_eval = X_eval.copy(deep=True)

    # Train Linear Regression model
    def train(self):
        svr_model = SVR(kernel='linear', )
        svr_model.fit(self.X_train, self.y_train)
        # Save the models
        joblib.dump(svr_model, "ML/svr_model.pkl") 

    def predict(self):
        svr_model = joblib.load("ML/svr_model.pkl")
        return svr_model.predict(self.X_eval)