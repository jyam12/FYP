import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import mean_squared_error
import joblib

class GradientBoosting:
    # Parse and preprocess relevant columns for all levels
    def __init__(self,X_train,y_train,X_eval):

        self.X_train = X_train.copy(deep=True)
        self.y_train = y_train.copy(deep=True)
        self.X_eval = X_eval.copy(deep=True)

    def findparameter(self):
        # Create base model
        gb_model = GradientBoostingRegressor(random_state=42)

        # Define parameter grid
        param_grid = {
            # Key parameters for price movement prediction
            'n_estimators': [100, 200, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': range(3, 12, 2),
            'min_samples_split': range(2, 12, 2),
            'min_samples_leaf': range(2, 12, 2),
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        # Configure HalvingGridSearchCV
        halving_search = HalvingGridSearchCV(
            estimator=gb_model,
            param_grid=param_grid,
            min_resources='exhaust',  # Start with small subset and gradually increase
            scoring='neg_mean_squared_error',
            n_jobs=-1,  # Use all available cores
            random_state=42,
            verbose=0  # Print progress
        )
        # Fit the model
        halving_search.fit(self.X_train, self.y_train)

        # Get results
        print("\nBest parameters:", halving_search.best_params_)
        print("Best score:", -halving_search.best_score_)  # Convert back from negative MSE

    # Train Linear Regression model
    def train(self):
        GB_model = GradientBoostingRegressor(
                    n_estimators=100,     # number of boosting stages
                    learning_rate=0.1,    # learning rate shrinks the contribution of each tree
                    max_depth=5,          # maximum depth of individual regression estimators
                    min_samples_split=1555,  # minimum samples required to split internal node
                    min_samples_leaf=5,   # minimum samples required to be at a leaf node
                    random_state=42
                )
        GB_model.fit(self.X_train, self.y_train)
        # Save the models
        joblib.dump(GB_model, "ML/GB_model_test.pkl") 

    def predict(self):
        GB_model = joblib.load("ML/GB_model_test.pkl")
        return GB_model.predict(self.X_eval)