import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("Airbnb-cleaned.csv")
df.columns
del df["Unnamed: 0"]

df1 = df[['neighbourhood', 'property_type', 'room_type']]
# IMPORT ENCODER
from sklearn.preprocessing import OneHotEncoder

# FIT ENCODER ON THE ORIGINAL DATASET TO MAKE IT REMEMBER CATEGORIES
enc = OneHotEncoder(sparse=False)
enc.fit(df1)



df["neighbourhood"].unique()

df[['Bijlmer-Oost', 'Noord-Oost', 'Noord-West', 'Oud-Noord',
         'IJburg - Zeeburgereiland', 'Centrum-West',
         'Oostelijk Havengebied - Indische Buurt', 'Centrum-Oost',
         'Oud-Oost', 'Watergraafsmeer', 'Gaasperdam - Driemond',
         'Westerpark', 'Bijlmer-Centrum', 'De Pijp - Rivierenbuurt', 'Zuid',
         'Buitenveldert - Zuidas', 'De Baarsjes - Oud-West',
         'Bos en Lommer', 'Geuzenveld - Slotermeer', 'Slotervaart',
         'Osdorp', 'De Aker - Nieuw Sloten',
         'Apartment', 'Bed & Breakfast', 'House',
         'Entire home/apt', 'Private room', 'Shared room']] = enc.transform(
       df1[["neighbourhood", "property_type", "room_type"]])


df = df.drop(["neighbourhood", "property_type", "room_type"], axis =1)
df["Distance_from_center(m)"] = df["Distance_from_center(m)"]/1000

y = df['price']
data = df.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=7)
model = RandomForestRegressor()
model.fit(X_train,y_train)
pred = model.predict(X_test)

mean_absolute_error(y_test, pred)

import pickle

filename = 'Airbnb.sav'
pickle.dump(model, open(filename, 'wb'))




