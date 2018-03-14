
from keras.layers import Conv1D, Activation, Dense, Input, Flatten, Add, Concatenate, Multiply
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.layers
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm



class AnomNet():

    def __init__(self, 
                 input_size = 256,
                 in_channels = 1,
                 num_filters = 16,
                 num_output_filter = 12,
                 use_skip = True,
                 look_ahead = 16,
                 lr_rate = 0.0002, 
                 epochs = 10):

        # training hyperparameters
        self.input_size = input_size
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_output_filter = num_output_filter
        self.use_skip = use_skip
        self.look_ahead = look_ahead

        self.lr_rate = lr_rate
        self.epochs = epochs

        # construct model
        self._construct_model()

    def _construct_model(self):
        inps = Input((self.input_size, self.in_channels))
        #Layer 1
        x = Conv1D(self.num_filters, 2, padding='causal', name='layer_1_causal_conv')(inps)
        skip_connections = []
        #We construct such that the final output's receptive field is equivalent to the input
        layers = int(np.log2(self.input_size//2) - 1)
        for layer in range(layers):
            x, skip_x = self._residual_block(x, self.num_filters, 2**(layer+1))
            skip_connections.append(skip_x)

        if self.use_skip:
            x = Add()(skip_connections)
        
        #We use Conv1D as a pseudo "Dense"
        x = Activation('relu')(x)
        x = Conv1D(self.num_output_filter, 1, activation='relu')(x)
        x = BatchNormalization(axis=2)(x)
        x = Conv1D(self.num_output_filter//2, 1, activation='relu')(x)
        x = BatchNormalization(axis=2)(x)
        x = Flatten()(x)

        #Perform regression
        x = Dense((self.look_ahead * self.in_channels)//2)(x)
        x = Dense(self.look_ahead * self.in_channels)(x)
        x = keras.layers.Reshape((self.look_ahead, self.in_channels))(x)
        model = Model(inputs=inps, outputs=x)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        self.model = model

    def _residual_block(self, x, nb_filters, dilation_rate):
        tanh_out = Conv1D(nb_filters, 2, dilation_rate=dilation_rate, padding='causal', 
                          activation='tanh')(x) #, kernel_regularizer=regularizers.l2(0.05)
        sigm_out = Conv1D(nb_filters, 2, dilation_rate=dilation_rate, padding='causal', 
                          activation='sigmoid')(x) #, kernel_regularizer=regularizers.l2(0.05)

        z = Multiply()([tanh_out, sigm_out])

        #Implementation v1
        skip_x = Conv1D(nb_filters, 1, padding='same')(z)
        res_x = Add()([skip_x, x]) #Residual connection

        return res_x, skip_x

    def _reshape_and_replicate(self, x, look_back, look_ahead):
        """Reshapes and replicate data into format convenient for training
        x: Shape (t_steps, channels)
        look_back: Number of time-steps network depends on
        look_ahead: Number of time-steps network attempts to predict ahead
        
        Output:
        X_in: History with size (t_len - look_ahead - look_back, look_back, channels)
        X_out: Prediction target with size (t_len - look_ahead - look_back, look_ahead, channels)
        """
        t_len = x.shape[0]
        channels = x.shape[1]
        X_input = np.zeros((t_len - look_ahead - look_back, look_back, channels))
        X_out = np.zeros((t_len - look_ahead - look_back, look_ahead, channels))
        for i in range(len(X_input)):
            X_input[i, :] = x[i:i+look_back, :].reshape(1, look_back, channels)
            X_out[i, :] = x[i+look_back:i+look_back+look_ahead, :].reshape(1, look_ahead, channels)
            
        return X_input, X_out

    def fit(self, x_raw):
        
        """Perform anomaly detection on time-series x
        x: np.array of dimension (, n_dim), 0-th index is time,
        and 1st index is for multi_dimension input"""
        self.single_dim = False
        if len(x_raw.shape) == 1:  # 1 dimensional time-series
            self.in_channels = 1
            x_raw = x_raw.reshape(-1, 1)
            self.single_dim = True
        elif len(x_raw.shape) == 2:  # Multi-dim time-series
            in_channels = x_raw.shape[1]
        else:
            raise(Exception("Unexpected input shape. Expected 1 or 2 dimension. Received: {}".format(len(x_raw.shape))))
        self.t_len = x_raw.shape[0]
        x_raw = x_raw.astype(np.float32)
        self.x_raw = x_raw
        
        #Perform normalization
        self.scaler = StandardScaler()
        #scaler = MinMaxScaler()
        x = self.scaler.fit_transform(x_raw)
        X_input, X_out = self._reshape_and_replicate(x, self.input_size, self.look_ahead)
        self.X_input = X_input
        self.model.fit(X_input, X_out, verbose=1, epochs=self.epochs)
        
        return self.model

    def predict(self, X_input):
        X_input = self.X_input
        pred = self.model.predict(X_input)
        
        #Predict
        #pred = model.predict(X_input)
        pred = self.scaler.inverse_transform(pred)    
        
        #Correc the time-frames
        pred_full = np.full((self.t_len, self.look_ahead, self.in_channels), np.nan, dtype=np.float32)
        for i in range(self.look_ahead):  # Shift the time-frames forward
            pred_full[self.input_size+i:self.t_len-self.look_ahead+i, i, :] = pred[:, i, :]
        #Truncate the extra predictions
        #pred_full = pred_full[:t_len , :, :]
        
        error = pred_full - self.x_raw.reshape((-1, 1, self.in_channels))
        pred_mean = np.nanmean(pred_full, axis=1)
        pred_dev = np.std(pred_full, axis=1)
        if self.single_dim:
            pred_full = pred_full.squeeze()
            error = error.squeeze()
            pred_mean = pred_mean.squeeze()
            
        output = {'pred_full': pred_full,
                  'pred': pred_mean,
                  'error': error,
                  'conf': pred_dev,
                  }
        return output


    def load(self, epoch):
        pass

