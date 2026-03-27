# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 23:09:35 2026

@author: Ahsan
"""

import yfinance as yf
import datetime

ticker = "AAPL"
# Fetch data for a significant period (e.g., 5-10 years) for better training
data = yf.download(ticker, start="2016-01-01", end="2026-02-23")


data['Target_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Target_Close']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print("done")