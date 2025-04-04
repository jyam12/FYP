# %%
from sklearn.metrics import mean_squared_error
import numpy as np

# %%
# Evaluate performance using MSE
def cal_mean_squared_error(predict: np.array, actual: np.array):
    mse=np.array([])
    if(predict.shape!=actual.shape):
        print("The length of predict and actural not match")
        return None 
    if (len(predict.shape) > 1):
        for i in range(predict.shape[1]):
            mse=np.append(mse,mean_squared_error(predict[:,i], actual[:,i]))
    else:
        mse=np.append(mse,mean_squared_error(predict, actual))
    return mse

# %%
# Evaluate performance using directional accuracy
def cal_directional_accuracy(predict: np.array, actual: np.array):
    directional_accuracy=np.array([])
    if(predict.shape!=actual.shape):
        print("The length of predict and actural not match")
        return None 
    if (len(predict.shape) > 1):
        for i in range(predict.shape[1]):
            directional_accuracy=np.append(directional_accuracy,(predict[:,i]==actual[:,i]).mean())
    else:
        directional_accuracy=np.append(directional_accuracy,(predict==actual).mean())
    return directional_accuracy

# %%
# Evaluate performance using Tolerance-Based accuracy
def cal_Tolerance_Based_accuracy(predict: np.array, actual: np.array):
    Tolerance_Based_accuracy=np.array([])
    if(predict.shape!=actual.shape):
        print("The length of predict and actural not match")
        return None 
    if (len(predict.shape) > 1):
        for i in range(predict.shape[1]):
            tolerance = 0.0001 * actual[:,i].mean()
            Tolerance_Based_accuracy=np.append(Tolerance_Based_accuracy,(abs(predict[:,i] - actual[:,i]) <= tolerance).mean())
    else:
        tolerance = 0.0001 * actual.mean()
        Tolerance_Based_accuracy=np.append(Tolerance_Based_accuracy,(abs(predict- actual) <= tolerance).mean())
    return Tolerance_Based_accuracy


