# --- MODEL AND DATA LOADING LOGIC (From deeplob_classification.py) ---

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os
import sys

# 1. DEEPLOB MODEL (MODIFIED FOR CLASSIFICATION)
class DeepLOBClassification:
    """
    DeepLOB model adapted for 3-class classification (Up, Down, Stationary)
    as required by the FI-2010 dataset.
    """
    
    def __init__(self, T=100, n_features=40, conv_filters=[32, 32, 32], 
                 lstm_units=64, dense_units=64, dropout_rate=0.2):
        self.T = T
        self.n_features = n_features
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        inputs = layers.Input(shape=(self.T, self.n_features, 1))
        
        # Convolutional Block 1
        x = layers.Conv2D(filters=self.conv_filters[0], kernel_size=(1, 2), strides=(1, 2))(inputs)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(filters=self.conv_filters[0], kernel_size=(4, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(filters=self.conv_filters[0], kernel_size=(4, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        
        # Convolutional Block 2
        x = layers.Conv2D(filters=self.conv_filters[1], kernel_size=(1, 2), strides=(1, 2))(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(filters=self.conv_filters[1], kernel_size=(4, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(filters=self.conv_filters[1], kernel_size=(4, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        
        # Convolutional Block 3
        x = layers.Conv2D(filters=self.conv_filters[2], kernel_size=(1, 10))(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(filters=self.conv_filters[2], kernel_size=(4, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Conv2D(filters=self.conv_filters[2], kernel_size=(4, 1), padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        
        # Reshape for LSTM
        x = layers.Reshape((self.T, -1))(x)
        
        # LSTM layer
        x = layers.LSTM(self.lstm_units)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layer
        x = layers.Dense(self.dense_units)(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer (Classification - 3 classes: Down, Stationary, Up)
        outputs = layers.Dense(3, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
    def summary(self):
        if self.model is None:
            self.build_model()
        self.model.summary()
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, verbose=1, callbacks=None):
        if self.model is None:
            self.compile_model()
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    def load_model(self, filepath):
        self.model = keras.models.load_model(
            filepath, 
            custom_objects={'LeakyReLU': layers.LeakyReLU}
        )

# 2. FI-2010 DATA LOADING LOGIC (from deeplob_classification.py)
def get_raw_data(auction, normalization, day):
    root_path = "/kaggle/input/lobdata/BenchmarkDatasets"
    if auction:
        path1 = "Auction"
    else:
        path1 = "NoAuction"

    if normalization == 'Zscore':
        tmp_path_1 = '1.'
        normalization_filename = 'ZScore'
    elif normalization == 'MinMax':
        tmp_path_1 = '2.'
        normalization_filename = normalization
    elif normalization == 'DecPre':
        tmp_path_1 = '3.'
        normalization_filename = normalization

    tmp_path_2 = f"{path1}_{normalization}"
    path2 = f"{tmp_path_1}{tmp_path_2}"

    if day == 1:
        path3 = tmp_path_2 + '_' + 'Training'
        filename = f"Train_Dst_{path1}_{normalization_filename}_CF_{str(day)}.txt"
    else:
        path3 = tmp_path_2 + '_' + 'Testing'
        day_idx = day - 1
        filename = f"Test_Dst_{path1}_{normalization_filename}_CF_{str(day_idx)}.txt"

    file_path = os.path.join(root_path, path1, path2, path3, filename)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None

    print(f"Loading data from: {filename}")
    fi2010_dataset = np.loadtxt(file_path)
    return fi2010_dataset

def extract_stock_data(raw_data, stock_idx):
    n_boundaries = 4
    boundaries = np.sort(
        np.argsort(np.abs(np.diff(raw_data[0], prepend=np.inf)))[-n_boundaries - 1:]
    )
    boundaries = np.append(boundaries, [raw_data.shape[1]])
    split_data = tuple(raw_data[:, boundaries[i] : boundaries[i + 1]] for i in range(n_boundaries + 1))
    return split_data[stock_idx]

def split_x_y(data, lighten):
    if lighten:
        data_length = 20
    else:
        data_length = 40

    x = data[:data_length, :].T
    y = data[-5:, :].T
    return x, y

def data_processing(x, y, T, k):
    [N, D] = x.shape

    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    y_proc = y[T - 1:N]
    y_proc = y_proc[:, k] 
    y_proc = y_proc - 1 
    
    return x_proc, y_proc

def load_fi2010_data(auction, normalization, stock_indices, days, T, k, lighten):
    x_cat = np.array([])
    y_cat = np.array([])
    
    for stock in stock_indices:
        print(f"\n--- Processing Stock Index: {stock} ---")
        for day in days:
            day_data_raw = get_raw_data(auction=auction, normalization=normalization, day=day)
            
            if day_data_raw is None:
                continue
                
            day_data_stock = extract_stock_data(day_data_raw, stock)
            
            x, y = split_x_y(day_data_stock, lighten)
            x_day, y_day = data_processing(x, y, T, k)

            if len(x_cat) == 0 and len(y_cat) == 0:
                x_cat = x_day
                y_cat = y_day
            else:
                x_cat = np.concatenate((x_cat, x_day), axis=0)
                y_cat = np.concatenate((y_cat, y_day), axis=0)

    print("\n--- Data Loading Complete ---")
    
    if len(x_cat) > 0:
        x_cat = np.expand_dims(x_cat, axis=-1)
    
    return x_cat, y_cat


# --- 3. TRAINING EXECUTION SCRIPT (New Parameters) ---

AUCTION = False             # Use "NoAuction" dataset
NORMALIZATION = "Zscore"    # Use "Zscore" normalization
STOCK_INDICES = [0, 1, 2, 3, 4] # Load all 5 stocks
LIGHTEN = False             # Use all 10 LOB levels (n_features=40)

# Model Parameters
T = 100            # 100 time steps per sample
N_LEVELS = 10      
N_FEATURES = N_LEVELS * 4  # 40 features

# Prediction Horizon Parameter
K = 0 # We will predict the 10-tick horizon (k=0)

# Training Parameters (UPDATED)
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.0001


# 4. EXECUTION
if __name__ == '__main__':
    # Load Training Data (Day 1)
    print("=== Loading Training Data (Day 1) ===")
    train_days = [1]
    X_train, y_train = load_fi2010_data(
        AUCTION, NORMALIZATION, STOCK_INDICES, train_days, T, K, LIGHTEN
    )

    # Load Testing Data (Days 2-10)
    print("\n=== Loading Testing Data (Days 2-10) ===")
    test_days = list(range(2, 11)) # Days 2 through 10
    X_test, y_test = load_fi2010_data(
        AUCTION, NORMALIZATION, STOCK_INDICES, test_days, T, K, LIGHTEN
    )

    if len(X_train) == 0 or len(X_test) == 0:
        print("\nError: Data loading failed. Exiting.")
        sys.exit()
        
    print(f"\nTotal training samples: {X_train.shape[0]}")
    print(f"\nTotal testing samples:  {X_test.shape[0]}")
    
    # CREATE AND COMPILE MODEL
    model = DeepLOBClassification(T=T, n_features=N_FEATURES)
    model.compile_model(learning_rate=LEARNING_RATE)

    print("\n--- Model Summary ---")
    model.summary()

    # Add a callback for early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # TRAIN THE MODEL
    print("\n--- Starting Model Training ---")
    history = model.train(
        X_train, y_train,
        X_val=X_test,  # Using the test set as the validation set
        y_val=y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    # EVALUATE THE MODEL
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # SAVE MODEL (optional)
    model_save_path = f'deeplob_classification_k{K}.keras'
    model.save_model(model_save_path)
    print(f"\nModel saved to {model_save_path}")

