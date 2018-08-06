from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np


class KddCupModel(object):
    pass
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
    def __init__(self, data, layers=[], model_path=None):
        """

        data: object of type data.Data
        layers: a list of dicts, each representing a layer
        """
        self.data = data
        if not model_path:
            self.model = self.__build_model(layers)
        else:
            self.model = load_model(model_path)

    def __build_model(self, layers):
        # layers = [{'neurons': neurons, 'activation': activation, ...}, 
                  # {'neurons': neurons, 'activation': activation, ...},
                  #  ... ]
        model = Sequential()
        model.add(Dense(layers[0]['neurons'], input_shape=(self.data.input_dim,), activation=layers[0]['activation']))
        for layer in layers[1:]:
            model.add(Dense(layer['neurons'], activation=layer['activation']))
        model.add(Dense(self.data.output_dim, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, epochs=10, verbose=1):
        data = self.data.binarized
        self.model.fit(data.X, data.Y, batch_size=128, epochs=epochs, verbose=verbose)

    def test(self, data):
        data = data.binarized
        loss, acc = self.model.evaluate(data.X, data.Y)
        return loss, acc

    def save(self, path):
        self.model.save(path)