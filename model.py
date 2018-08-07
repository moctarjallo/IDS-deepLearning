from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np


# class KddCupModel(object):
#     def __init__(self, inputs=[], targets=['normal.', 'other.'], layers=[]):
#         self.callbacks = [TensorBoard(log_dir='../logs/tensorboard/15'), ModelCheckpoint(filepath='../logs/models/model-last.h5')]
#         self.inputs = inputs
#         self.targets = targets
#         self.layers = layers
#         if inputs:
#             input_dim = np.array(inputs).shape[0]
#         else:
#             input_dim = 41
#         self.model = self.__build_model(input_dim)

#     def __build_model(self, input_dim):
#         # layers = [{'neurons': neurons, 'activation': activation, ...}, 
#                   # {'neurons': neurons, 'activation': activation, ...},
#                   #  ... ]
#         output_shape = np.array(self.targets).shape
#         model = Sequential()
#         model.add(Dense(self.layers[0]['neurons'], input_shape=(input_dim,), activation=self.layers[0]['activation']))
#         for layer in self.layers[1:]:
#             model.add(Dense(layer['neurons'], activation=layer['activation']))
#         model.add(Dense(output_shape[0], activation='softmax'))
#         model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         return model

#     def train(self, data, batch_size=128, epochs=10, verbose=1):
#         d = next(data)[self.inputs + self.targets].binarized
#         self.model.fit(x=d.X, y=d.Y, batch_size=batch_size, epochs=10, verbose=verbose)

#     def test(self, data, verbose=1):
#         d = next(data)[self.inputs + self.targets].binarized
#         loss, acc = self.model.evaluate(x=d.X, y=d.Y, verbose=verbose)
#         return round(loss, 4), round(acc, 4)


class Model(object):
    def __init__(self, data, layers=[], model=None, model_path=None):
        """

        data: object of type data.Data
        layers: a list of dicts, each representing a layer
        """
        self.data = data.binarized
        if model:
            self.model = model
        elif model_path:
            self.model = load_model(model_path)
        else:
            self.model = self.__build_model(layers)

    @property
    def input_dim(self):
        return self.data.shape[1] - 1 # not include the output dim

    @property
    def output_dim(self):
        return len(set(self.data.attack_types))

    def __build_model(self, layers):
        # layers = [{'neurons': neurons, 'activation': activation, ...}, 
                  # {'neurons': neurons, 'activation': activation, ...},
                  #  ... ]
        model = Sequential()
        model.add(Dense(layers[0]['neurons'], input_shape=(self.input_dim,), activation=layers[0]['activation']))
        for layer in layers[1:]:
            model.add(Dense(layer['neurons'], activation=layer['activation']))
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, epochs=10, verbose=1):
        self.model.fit(self.data.X, self.data.Y, batch_size=128, epochs=epochs, verbose=verbose)

    def test(self, data=None):
        if data:
            data = data.binarized
        else:
            data = self.data
        loss, acc = self.model.evaluate(data.X, data.Y)
        return loss, acc

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train_then(self, epochs=10, verbose=0):
        self.train(epochs=epochs, verbose=verbose)
        return self

    def save(self, path):
        self.model.save(path)

    
class KddCupModel(object):
    def __init__(self, data):
        """

        data: object of type data.KddCupData
        """
        self.data = data
    
    def train(self, inputs=[], targets=[], layers=[]):
        for d in self.data:
            mdl = Model(d[inputs][targets], layers=layers)
            if mdl.output_dim == len(targets):
                mdl.train()
            else:
                print('Skipping..')

    def test(self, data):
        pass