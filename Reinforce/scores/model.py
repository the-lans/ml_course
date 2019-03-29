import time
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.layers.recurrent import LSTM

LAYERS = [24, 24] #Количество нейронов в скрытых слоях
ACTIV = ["tanh", "tanh", "linear"] #Функции активации слоёв
TYPE_NET = "dense" #Тип нейросети

class ModelNeuro:
    def __init__(self, env, layers, activ, typenet, lr):
        self.model = None
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.timesteps = env.timesteps
        self.lr = lr
        self.drop = 0.2
        self.rnn_drop = 0.0
        self.typenet = typenet
        layers.insert(0, env.observation_space)
        layers.append(env.action_space)
        self.model_compile(layers, activ)

    def model_compile(self, layers, activ):
        if self.typenet == "dense":
            self.model_compile_dense(layers, activ)
        elif self.typenet == "lstm":
            self.model_compile_lstm(layers, self.timesteps, self.drop, self.rnn_drop)

    def model_compile_dense(self, layers, activ):
        self.model = Sequential()
        self.model.add(Dense(layers[1], input_shape=(layers[0],), activation=activ[0]))
        ind = 2
        while ind < len(layers) - 1:
            self.model.add(Dense(layers[ind], activation=activ[ind-1]))
            ind += 1
        self.model.add(Dense(layers[ind], activation=activ[ind-1]))
        start = time.time()
        self.model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.lr))
        # self.model.compile(loss="mse", optimizer=optimizers.SGD(lr=0.1))
        print('compilation time: ', time.time() - start)
        print(self.model.summary())

    def model_compile_lstm(self, layers, timesteps, drop, rnn_drop, activ = "linear"):
        self.model = Sequential()
        self.model.add(LSTM(layers[1], input_shape=(timesteps, layers[0]), recurrent_dropout=rnn_drop, return_sequences=True))
        # model.add(LSTM(input_dim=data_dim, output_dim=timesteps, return_sequences=True))
        self.model.add(Dropout(drop))
        ind = 2
        while ind < len(layers)-1:
            self.model.add(LSTM(layers[ind], recurrent_dropout=rnn_drop, return_sequences=False))
            self.model.add(Dropout(drop))
            ind += 1
        self.model.add(Dense(layers[ind], activation=activ))
        start = time.time()
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        print('compilation time: ', time.time() - start)
        print(self.model.summary())

    def predict(self, state):
        return self.model.predict(state)

    def fit(self, state, q_values):
        return self.model.fit(state, q_values, verbose=0)
