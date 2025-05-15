import os
import sys

import numpy as np
import pandas as pd
import dill

from exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obejcts(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,x_test,y_train,y_test,models:dict,params):
    try:
        report = {}
        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            grid_search = GridSearchCV(estimator=model,param_grid=param,cv=3,scoring='r2',refit=True)
            grid_search.fit(x_train,y_train)
            # model.fit(x_train,y_train)
            model.set_params(**grid_search.best_params_)
            model.fit(x_train,y_train)
            y_test_pred = model.predict(x_test)
            test_model_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e,sys)
