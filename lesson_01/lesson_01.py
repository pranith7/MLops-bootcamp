# 01_introduction 

import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import joblib

def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df




def process_and_predict_duration(df_train, df_val):
    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']
    
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    mse = mean_squared_error(y_val, y_pred, squared=False)
    # joblib.dump(lr, 'models/linear_regression_model.bin')

    # # with open('models/linear_regression_model.bin', 'wb') as f_out:
    # #     pickle.dump((dv, lr), f_out)
    
    # lr = Lasso(0.01)
    # lr.fit(X_train, y_train)

    # y_pred = lr.predict(X_val)

    # mse = mean_squared_error(y_val, y_pred, squared=False)

    return mse


df_train    = read_dataframe('/workspaces/MLops-bootcamp/lesson_01/data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('/workspaces/MLops-bootcamp/lesson_01/data/green_tripdata_2021-02.parquet')

print(f"Length of train and valid dataframe --> {len(df_train)}, {len(df_val)}")
mse_result = process_and_predict_duration(df_train,df_val)
print(f"Mean Squared Error: {mse_result}")







