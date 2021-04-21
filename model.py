import pandas as pd
#import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('/home/swanand/Desktop/Python Projects/Car Price Prediction/Car details v3.csv')
df.dropna(inplace=True)
df['mileage'] = df['mileage'].apply(lambda x: float(x.split()[0]))
df['engine'] = df['engine'].apply(lambda x: float(x.split()[0]))
df['max_power'] = df['max_power'].apply(lambda x: float(x.split()[0]))
df.drop(['torque'], axis=1, inplace=True)
df.drop(['max_power'], axis=1, inplace=True)
df['age'] = 2020 - df['year']
df.drop(['year'], axis=1, inplace=True)
df['owner'].replace('First Owner', 1, inplace=True)
df['owner'].replace('Second Owner', 2, inplace=True)
df['owner'].replace('Third Owner', 3, inplace=True)
df['owner'].replace('Fourth & Above Owner', 4, inplace=True)
df.drop(df[df['owner'] == 'Test Drive Car'].index, inplace=True)
df['owner'] = df['owner'].apply(lambda x: int(x))
df.drop(['name'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)
df.rename(columns={"seller_type_Trustmark Dealer": "seller_type_Trustmark_Dealer"}, inplace=True)

X = df.iloc[:, 1:]
y = df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)


def predict_price(km_driven, no_of_owners, mileage, engine, seats, age, fuel_type, seller_type, transmission):
    if fuel_type == 'CNG':
        fuel_Diesel, fuel_LPG, fuel_Petrol = 0, 0, 0
    elif fuel_type == 'Diesel':
        fuel_Diesel = 1
        fuel_LPG, fuel_Petrol = 0, 0
    elif fuel_type == 'LPG':
        fuel_LPG = 1
        fuel_Diesel, fuel_Petrol = 0, 0
    else:
        fuel_Petrol = 1
        fuel_LPG, fuel_Diesel = 0, 0
    
    if seller_type == 'Dealer':
        seller_type_Individual , seller_type_Trustmark_Dealer = 0, 0
    elif seller_type == 'Individual':
        seller_type_Individual = 1
        seller_type_Trustmark_Dealer = 0
    else:
        seller_type_Trustmark_Dealer = 1
        seller_type_Individual = 0
    
    if transmission == 'Manual':
        transmission_Manual = 1
    else:
        transmission_Manual = 0

    pred = rf.predict([[km_driven, no_of_owners, mileage, engine, seats, age, fuel_Diesel, fuel_LPG, fuel_Petrol, seller_type_Individual, seller_type_Trustmark_Dealer, transmission_Manual ]])
    pred = np.array_str(pred)
    return round(float(pred[1:-2]))


#pickle.dump(rf, open('/home/swanand/Desktop/Python Projects/Car Price Prediction/model.pkl', 'wb'))
