import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report
import time


from sklearn.preprocessing import LabelEncoder

# vidimo da treba ostaviti svojstva minutes i months
# weather description treba odbaciti. WeatherMain moze i ne mora ostati, ali bolje da ostane jer ce mozda
# imati vecu vaznost kad dobijemo jos podataka iz razlicitih dijelova godine
# ne vidi se razlika u performansama ako se koriste LabelEncoder i OneHot


def get_df(path):
    df = pd.read_csv(path)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['Minute'] = df['Timestamp'].dt.minute
    df['Year'] = df['Timestamp'].dt.year
    # one_hot = pd.get_dummies(df['WeatherMain'])
    # df = df.drop(['Unnamed: 0', 'Timestamp', 'WeatherDescription', 'WeatherMain'], axis=1)
    # df = df.join(one_hot)
    label_encoder = LabelEncoder()
    df['WeatherMain'] = label_encoder.fit_transform(df['WeatherMain'])
    df['WeatherDescription'] = label_encoder.fit_transform(df['WeatherDescription'])
    df = df.drop(['Unnamed: 0', 'Timestamp', 'index'], axis=1)
    return df


start_time = time.time()

path = '../../train.csv'
df = get_df(path)
X = df.drop('Detections', axis=1)
y = df['Detections']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)



mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')



feature_importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print('\nFeature Importance:')
print(feature_importance_df)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")






