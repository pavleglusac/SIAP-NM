import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
import time


start_time = time.time()


def get_df(path):
    df = pd.read_csv(path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Minute'] = df['Timestamp'].dt.minute
    df['Year'] = df['Timestamp'].dt.year
    one_hot = pd.get_dummies(df['WeatherMain'])
    df = df.drop(['index', 'Timestamp', 'WeatherDescription', 'WeatherMain'], axis=1)
    df = df.join(one_hot)
    return df


path = '../../train.csv'
df = get_df(path)
X = df.drop('Detections', axis=1)
y = df['Detections']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

rf_model = RandomForestRegressor(n_estimators=70, random_state=42, bootstrap=True, ccp_alpha=0.013, max_samples=0.5)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Vrijeme: {elapsed_time} s")
