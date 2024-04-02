import numpy as np
import pandas as pd

from keras.layers import Dense, LSTM, GRU
from keras.models import Sequential
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras_tuner.tuners import BayesianOptimization


def df_to_X_y_vector(df, lags, future):
    X = []
    y = []
    for i in range(len(df) - lags - future):
        x_window_rows = df.iloc[i:i + lags]
        x_row = [list(x_window_rows.iloc[j]) for j in range(lags)]
        X.append(x_row)
        y_window_rows = df.iloc[i + lags:i + lags + future]
        y_row = [y_window_rows.iloc[j]['Detections'] for j in range(future)]
        y.append(y_row)
    return np.array(X), np.array(y)


def df_to_X_y_standard(df, lags):
    X = []
    y = []
    for i in range(len(df) - lags):
        window_rows = df.iloc[i:i + lags]
        row = [list(window_rows.iloc[j]) for j in range(lags)]
        X.append(row)
        label = df.iloc[i + lags]
        y.append(label['Detections'])
    return np.array(X), np.array(y)


def get_df(path):
    df = pd.read_csv(path)
    df = df.drop(['WeatherDescription', 'Unnamed: 0', 'index'], axis=1)
    df = pd.get_dummies(df, columns=['WeatherMain'], prefix='WeatherMain')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Year'] = df['Timestamp'].dt.year
    df['Month'] = df['Timestamp'].dt.month
    df['Day'] = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour
    df = df.drop(['Timestamp', 'WeatherMain_Snow', 'Year'], axis=1)
    return df_to_X_y_vector(df, 288, 288)


def build_model_lstm(hp):
    model = Sequential()
    for i in range(hp.Int('layers', 0, 10)):
        model.add(LSTM(
            units=hp.Int('units_' + str(i), min_value=50, max_value=100, step=10),
            return_sequences=True,
            # return_state=True,
            activation=hp.Choice('activation_' + str(i), ['relu', 'sigmoid', 'tanh']),
            dropout=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1),
            recurrent_dropout=hp.Float('recurrent_dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1),
            kernel_initializer=hp.Choice('kernel_initializer_' + str(i), ['glorot_uniform', 'orthogonal']),
            recurrent_initializer=hp.Choice('recurrent_initializer_' + str(i), ['orthogonal', 'glorot_uniform']),
            bias_initializer=hp.Choice('bias_initializer_' + str(i), ['zeros', 'ones']),
            input_shape=(X_train.shape[1], X_train.shape[2]),
        ))
    model.add(LSTM(units=hp.Int('units_last', min_value=50, max_value=100, step=10),
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['mse']
    )
    return model



def build_model_gru(hp):
    model = Sequential()
    for i in range(hp.Int('layers', 0, 10)):
        model.add(GRU(
            units=hp.Int('units_' + str(i), min_value=50, max_value=100, step=10),
            return_sequences=True,
            activation=hp.Choice('activation_' + str(i), ['relu', 'sigmoid', 'tanh']),
            dropout=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1),
            recurrent_dropout=hp.Float('recurrent_dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1),
            kernel_initializer=hp.Choice('kernel_initializer_' + str(i), ['glorot_uniform', 'orthogonal']),
            recurrent_initializer=hp.Choice('recurrent_initializer_' + str(i), ['orthogonal', 'glorot_uniform']),
            bias_initializer=hp.Choice('bias_initializer_' + str(i), ['zeros', 'ones']),
            input_shape=(X_train.shape[1], X_train.shape[2]),
        ))
    model.add(GRU(units=hp.Int('units_last', min_value=50, max_value=100, step=10),
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['mse']
    )
    return model


path = '../../train_belgrade.csv'
X, y = get_df(path)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(X_train.shape, y_train.shape)

tuner = BayesianOptimization(
    build_model_gru,
    objective='val_loss',
    num_initial_points=3,
    max_trials=3,
    directory='automl',
    project_name='bayesian_gru'
)
tuner.search(X_train, y_train, epochs=15, validation_data=(X_val, y_val), batch_size=256)
best_model = tuner.get_best_models()[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

for param in best_hps.values:
    print(f"{param}: {best_hps.get(param)}")

best_model.fit(X_train, y_train, epochs=15, batch_size=256)
test_score = best_model.evaluate(X_test, y_test)
print('test score je: ', test_score)

best_model.save('best_model.h5')
