import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations
from keras import callbacks
import joblib



class NeuralNetwork:
    def __init__(self, df):
        self.df = df
        self.scaler_input = None
        self.scaler_output = None
        self.model = None
        self.scaler_input_path = "scaler_input.pkl" 
        self.scaler_output_path = "scaler_output.pkl"
        self.model_path = "model.h5"  
        
    def _process_Xy(self, raw_X: np.array, raw_y: np.array, lookback: int) -> np.array:
        X = np.empty(shape=(raw_X.shape[0] - lookback, lookback , raw_X.shape[1]), dtype=np.float32)
        y = np.empty(shape=(raw_y.shape[0] - lookback), dtype=np.float32)

        target_index = 0
        for i in range(lookback, raw_X.shape[0]):
            X[target_index] = raw_X[i - lookback : i]
            y[target_index] = raw_y[i]
            target_index += 1

        return X.copy(), y.copy()
        
    def fit_model(self):
        plt.figure(figsize=(14, 6))
        plt.plot(self.df.index, self.df['high'], label='High')
        plt.plot(self.df.index, self.df['low'], label='Low')
        plt.plot(self.df.index, self.df['open'], label='Open')
        plt.plot(self.df.index, self.df['close'], label='Close')
        plt.title('Time Series - BTC Prices')
        plt.legend()
        plt.savefig('btc_prices.png')
        

        self.df.set_index('datetime', inplace=True)
        self.df["Day.Of.Year.X"] = np.sin(2 * np.pi * self.df.index.day_of_year / 365)
        self.df["Day.Of.Year.Y"] = np.cos(2 * np.pi * self.df.index.day_of_year / 365)
        # self.df = self.df.drop(["low", "close", "volume"], axis=1)
        self.df = self.df.drop(["date", 'week', 'month'], axis=1)
        
        plt.figure(figsize=(6, 6))
        plt.scatter(self.df['Day.Of.Year.X'], self.df['Day.Of.Year.Y'], c=self.df['high'])
        plt.colorbar(label='High Price')
        plt.title('Seasonal Effects on High Prices')
        plt.xlabel('Day.Of.Year.X')
        plt.ylabel('Day.Of.Year.Y')
        plt.savefig('seasonal_effects.png')
        

        
        train_split_index = int(len(self.df) * 0.7)   # 70% for training
        val_split_index = int(len(self.df) * 0.85)    # 15% for validation
        test_split_index = len(self.df)               # Remaining 15% for testing

        # Step 2: Split the DataFrame
        train_df = self.df.iloc[:train_split_index]   # Training set
        valid_df = self.df.iloc[train_split_index:val_split_index]   # Validation set
        test_df = self.df.iloc[val_split_index:test_split_index] 
        
        scaler_input = StandardScaler()
        scaler_output = StandardScaler()
        scaled_train = scaler_input.fit_transform(train_df)
        target_train = scaler_output.fit_transform(train_df[["high"]])
        scaled_valid = scaler_input.transform(valid_df)
        target_valid = scaler_output.transform(valid_df[["high"]])
        scaled_test = scaler_input.transform(test_df)
        target_test = scaler_output.transform(test_df[["high"]])
        
        lookback = 10
        train_X, train_y = self._process_Xy(scaled_train, target_train, lookback=lookback)
        test_X, test_y = self._process_Xy(scaled_test, target_test, lookback=lookback)
        valid_X, valid_y = self._process_Xy(scaled_valid, target_valid, lookback=lookback)
        
        model = keras.Sequential(
            [
                layers.LSTM(16, activation="relu", input_shape = train_X.shape[1:]),
                layers.Dense(1),
            ]
        )
        model.compile(loss='MeanSquaredError', optimizer='Adam')
        callbacks_value = [callbacks.EarlyStopping(monitor="val_loss", patience=10)]
        history = model.fit(
            train_X,
            train_y,
            validation_data=(valid_X, valid_y),
            batch_size=16,
            epochs=100,
            callbacks=callbacks_value,
            shuffle=True,
            verbose=True,
        )
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        
        pred = model.predict(test_X)
        plt.figure(figsize=(10, 6))
        plt.plot(test_df["high"], label="Real", color='blue')
        plt.plot(pd.DataFrame(index=test_df.index[lookback:], data=scaler_output.inverse_transform(pred)), label="Predicted", color='red')
        plt.xticks(rotation=45)
        plt.title('Actual vs Predicted high')
        plt.xlabel('Date')
        plt.ylabel('high')
        plt.legend()
        plt.savefig('prediction_plot.png')
        
        # Save the trained model and scaler

        
        self.scaler_input = scaler_input
        self.scaler_output = scaler_output
        self.model = model
        
        model.save(self.model_path)
        joblib.dump(self.scaler_input, self.scaler_input_path)
        joblib.dump(self.scaler_output, self.scaler_output_path)
        
        return model
    
    def load_model(scaler_path, model_path):
        scaler = joblib.load(scaler_path)
        model = keras.models.load_model(model_path)
        return scaler, model


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    neural_net = NeuralNetwork(df)
    trained_model = neural_net.fit_model()
    loaded_scaler, loaded_model = NeuralNetwork.load_model("scaler.pkl", "model.h5")