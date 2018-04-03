import pandas as pd
from datetime import datetime
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
df = pd.read_csv("sphist.csv")

# Convert the Date column to a Pandas date type
df["Date"] = pd.to_datetime(df["Date"])

#a Boolean series that tells you if each item in the Date column is after 2015-04-01
after = df["Date"] > datetime(year=2015, month=4, day=1)
df = df.sort_values(by = ["Date"], ascending=True)

# 3 indicators to compute, and generate a different column for each one
df['day_5'] = 0
df['day_30'] = 0
df['day_365'] = 0

df['day_5'] = pd.rolling_mean(df.Close, 5).shift()
df['day_30'] = pd.rolling_mean(df.Close, 30).shift()
df['day_365'] = pd.rolling_mean(df.Close, 365).shift()

# The dataset starts on 1950-01-03. Any rows that fall before 1951-01-03 don't have enough historical data to compute  the indicators use 365 days of historical data .Remove any rows from the DataFrame that fall before 1951-01-03
df = df[df["Date"] > datetime(year=1951, month=1, day=2)]

#remove any rows with NaN values
df = df.dropna(axis=0)
train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
test = df[df["Date"] > datetime(year=2013, month=1, day=1)]

#Initialize an instance of the LinearRegression class
lr = linear_model.LinearRegression()
lr.fit(train[['day_5','day_30','day_365']],train["Close"])
predictions = lr.predict(test[['day_5','day_30','day_365']])
mae = mean_absolute_error(test[['Close']], predictions)
mse = mean_squared_error(test['Close'],predictions)


