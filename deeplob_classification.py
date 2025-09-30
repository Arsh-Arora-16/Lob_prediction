import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os
import sys

# ---
# 1. DEEPLOB MODEL (MODIFIED FOR CLASSIFICATION)
# ---

class DeepLOBClassification:
    """
    DeepLOB model adapted for 3-class classification (Up, Down, Stationary)
    as required by the FI-2010 dataset.
    """
    
    def __init__(self, T=100, n_features=40, conv_filters=[32, 32, 32], 
                 lstm_units=64, dense_units=64, dropout_rate=0.2):
        """
        Args:
            T: Number of time steps (sequence length)
            n_features: Number of LOB features (10 levels * 4 features)
            conv_filters: List of filter numbers for each conv block
            lstm_units: Number of LSTM units
            dense_units: Number of units in dense layer
            dropout_rate: Dropout rate
        """
        self.T = T
        self.n_features = n_features
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self):
        """Build the DeepLOB classification model"""
        
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
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy', # For integer labels (0, 1, 2)
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
        # Need to provide LeakyReLU as a custom object
        self.model = keras.models.load_model(
            filepath, 
            custom_objects={'LeakyReLU': layers.LeakyReLU}
        )

# ---
# 2. FI-2010 DATA LOADING LOGIC (from GitHub file)
# ---

def get_raw_data(auction, normalization, day):
    """
    Handling function for loading raw FI2010 dataset
    MODIFIED to use the specific Kaggle path.
    """
    
    # HARDCODED root path for Kaggle environment
    root_path = "/kaggle/input/lobdata/BenchmarkDatasets"
    dataset_path = 'fi2010' # This part seems redundant with the full path

    if auction:
        path1 = "Auction"
    else:
        path1 = "NoAuction"

    if normalization == 'Zscore':
        tmp_path_1 = '1.'
    elif normalization == 'MinMax':
        tmp_path_1 = '2.'
    elif normalization == 'DecPre':
        tmp_path_1 = '3.'

    tmp_path_2 = f"{path1}_{normalization}"
    path2 = f"{tmp_path_1}{tmp_path_2}"

    if normalization == 'Zscore':
        normalization_filename = 'ZScore'
    else:
        normalization_filename = normalization


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
    """
    Extract specific stock data from raw FI2010 dataset
    """
    n_boundaries = 4
    boundaries = np.sort(
        np.argsort(np.abs(np.diff(raw_data[0], prepend=np.inf)))[-n_boundaries - 1:]
    )
    boundaries = np.append(boundaries, [raw_data.shape[1]])
    split_data = tuple(raw_data[:, boundaries[i] : boundaries[i + 1]] for i in range(n_boundaries + 1))
    return split_data[stock_idx]


def split_x_y(data, lighten):
    """
    Extract lob data and annotated label from fi-2010 data
    """
    if lighten:
        data_length = 20 # 5 levels * 4 features
    else:
        data_length = 40 # 10 levels * 4 features

    x = data[:data_length, :].T
    y = data[-5:, :].T # Last 5 rows are labels
    return x, y


def data_processing(x, y, T, k):
    """
    Process whole time-series-data
    """
    [N, D] = x.shape

    # x processing (create sequences)
    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    # y processing
    y_proc = y[T - 1:N]
    
    # Select label for prediction horizon 'k'
    # k=0 -> 10 ticks, k=1 -> 20 ticks, ..., k=4 -> 100 ticks
    y_proc = y_proc[:, k] 
    
    # Convert labels from (1, 2, 3) to (0, 1, 2)
    y_proc = y_proc - 1 
    
    return x_proc, y_proc

def load_fi2010_data(auction, normalization, stock_indices, days, T, k, lighten):
    """
    Main function to load and concatenate data from multiple days and stocks.
    Based on __init_dataset__ from the GitHub file.
    """
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
    
    # Reshape X for Conv2D input: (n_samples, T, n_features, 1)
    if len(x_cat) > 0:
        x_cat = np.expand_dims(x_cat, axis=-1)
    
    return x_cat, y_cat
