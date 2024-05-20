from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import json
import talib
import os
import env
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import neptune
import matplotlib.pyplot as plt

def get_previous_business_day(date):
    if date.weekday() == 0:  
        return date - timedelta(days=3)
    elif date.weekday() == 6:  
        return date - timedelta(days=2)
    else:
        return date - timedelta(days=1)

def get_years_ago_from_date(date, num_years):
    years_ago = date.replace(year=date.year - num_years)
    return get_previous_business_day(years_ago)

current_time = datetime.now()
end_date = get_previous_business_day(current_time)
start_date = get_years_ago_from_date(end_date, num_years=25)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

symbol = "BAYRY"

url = f"https://data.alpaca.markets/v2/stocks/bars?symbols={symbol}&timeframe=1Day&start={start_date_str}&end={end_date_str}&limit=10000&adjustment=raw&feed=sip&sort=asc"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": os.getenv("api-id"),  # Use environment variable
    "APCA-API-SECRET-KEY": os.getenv("api-secret"),  # Use environment variable
}

response = requests.get(url, headers=headers)

data = response.json()

symbol = list(data['bars'].keys())[0]
df = pd.DataFrame(data['bars'][symbol]) 

# Rename columns 
df.columns = ['y', 'High', 'Low', 'Number of Trades', 'Open', 'date', 'Volume', 'VWAP']

df['ds'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

from neuralprophet import NeuralProphet

df = df[['ds', 'y']]

m = NeuralProphet()
m.set_plotting_backend("plotly-static")  # show plots correctly in jupyter notebooks
metrics = m.fit(df)

predicted = m.predict(df)
forecast = m.predict(df)

m.plot(forecast)
