import numpy as np
import pandas as pd

# Importing the datasets
dataset = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")
ds = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

# Cleaning the data
# Dealing with missing latitude & longitude values for Aruba by directly importing those values
# Filling in missing province/state with the corresponding country.

dataset['Province/State'] = dataset['Province/State'].fillna(dataset['Country/Region'])
ds['Province/State'] = ds['Province/State'].fillna(ds['Country/Region'])

dataset['Lat'] = dataset['Lat'].fillna(value = 12)
dataset['Long'] = dataset['Long'].fillna(value = 70)

k = dataset[dataset['Lat'].isna()]  # Checking which country is missing latitude & longitude values- Aruba
ds['Lat'] = ds['Lat'].fillna(value = 12)
ds['Long'] = ds['Long'].fillna(value = 70)

# Separating Day & Month into individual columns
from datetime import datetime
month = lambda x: datetime.strptime(x, "%Y-%m-%d").month
day = lambda x: datetime.strptime(x, "%Y-%m-%d").day

dataset['Month'] = dataset['Date'].map(month)
dataset['Day'] = dataset['Date'].map(day)

ds['Month'] = ds['Date'].map(month)
ds['Day'] = ds['Date'].map(day)

# Finally dividing the dataset into matrix of independent features - X
# and matrix of dependent variables - y
X = dataset.iloc[:,[1, 2, 3, 4, 8, 9]].values
y = dataset.iloc[:,[6, 7]].values

Xx = ds.iloc[:,[1, 2, 3, 4, 6, 7]].values

# One Hot Encoding the country column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
lbl = LabelEncoder()
X[:, 0] = lbl.fit_transform(X[:, 0])
X[:, 1] = lbl.fit_transform(X[:, 1])

Xx[:, 0] = lbl.fit_transform(Xx[:, 0])
Xx[:, 1] = lbl.fit_transform(Xx[:, 1])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0, 1])],   
    remainder='passthrough'                                         
)
X = ct.fit_transform(X).toarray()
#X = np.delete(X, [273, 437], axis = 1)
Xx = ct.fit_transform(Xx).toarray()
#Xx = np.delete(Xx, [273, 437], axis = 1)

# Scaling - Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, :439] = sc.fit_transform(X[:, :439]) # Scaling applied to all columns except the last 2 columns- Day & Month
Xx[:, :439] = sc.transform(Xx[:, :439])

# Fitting the Decision Tree Regression Model on the training dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 1)
regressor.fit(X, y)

# Predictions on the test dataset
y_pred = pd.DataFrame(regressor.predict(Xx))
y_pred.iloc[:, 0] = [int(x) for x in y_pred.iloc[:, 0]]
y_pred.iloc[:, 1] = [int(x) for x in y_pred.iloc[:, 1]]


# Kaggle Submission CSV
pd.DataFrame({"ForecastId":list(range(1,len(ds)+1)),
              "ConfirmedCases":y_pred.iloc[:,0],
              "Fatalities": y_pred.iloc[:,1]}).to_csv("submission.csv",
                                           index = False,
                                           header = True)
