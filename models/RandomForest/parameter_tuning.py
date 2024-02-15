from scipy.stats import randint, uniform

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': randint(30, 200),
    # 'max_depth': randint(1, 20),
    # 'min_samples_split': randint(2, 20),
    # 'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False],
    'max_features': ['auto', 'sqrt', 'log2', None],  # Different options for feature selection
    # 'max_leaf_nodes': randint(2, 50),  # Maximum number of leaf nodes
    # 'min_impurity_decrease': uniform(0.0, 0.2),  # Minimum impurity decrease for a split
    'ccp_alpha': uniform(0.0, 0.2),  # Complexity parameter used for Minimal Cost-Complexity Pruning
    'max_samples': [None, 0.5, 0.75, 1.0]  # Fraction of samples used for fitting the base learners
}

rf_model = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(
    rf_model, param_distributions=param_dist, n_iter=3, cv=5, random_state=42, scoring='neg_mean_squared_error',
        verbose=2
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)
best_rf_model = random_search.best_estimator_
y_pred = best_rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)