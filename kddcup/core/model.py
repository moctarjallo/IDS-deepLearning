from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

from kddcup.core.constants import kddcup_targets

import numpy as np

import os

import json


class Model(object):
    def __init__(self, data=None, layers=[], kmodel=None):
        """

        data: object of type data.Data
        layers: a list of dicts, each representing a layer
        kmodel: object of type of keras.models.Sequential
        """
        if data:
            self.data = data.binarized
        if kmodel:
            self.kmodel = kmodel
        elif layers:
            self.kmodel = self.__build_model(layers)
        else:
            self.kmodel = None
        self.loss = -1
        self.accuracy = -2

    def __getitem__(self, *args):
        if isinstance(*args, str):
            return self.__dict__[str(*args)]
        keys = list(*args)
        return [self.__dict__[key] for key in keys]

    @property
    def input_dim(self):
        return self.data.shape[1] - 1  # not include the output dim

    @property
    def output_dim(self):
        return len(set(self.data.attack_types))

    def __build_model(self, layers):
        # layers = [{'neurons': neurons, 'activation': activation, ...},
                  # {'neurons': neurons, 'activation': activation, ...},
                  #  ... ]
        model = Sequential()
        model.add(Dense(layers[0]['neurons'], input_shape=(
            self.input_dim,), activation=layers[0]['activation']))
        for layer in layers[1:]:
            model.add(Dense(layer['neurons'], activation=layer['activation']))
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, batch_size=128, epochs=10, verbose=1):
        self.kmodel.fit(self.data.X, self.data.Y,
                        batch_size=batch_size, epochs=epochs, verbose=verbose)
        return self

    def test(self, data=None):
        if data:
            data = data.binarized
        else:
            data = self.data
        self.loss, self.accuracy = self.kmodel.evaluate(data.X, data.Y)
        return self

    def save(self, file):
        self.kmodel.save(file)
        return self

    def load(self, file):
        self.kmodel = load_model(file)
        return self


class KddCupModel(object):
    def __init__(self, inputs=[], targets=[], layers=[{'neurons': 1, 'activation': 'relu'}],
                 model_path=None):
        """Initialize self.model and set/get inputs and targets

        @params inputs: list of input properties to consider; default to [] 
                        meaning all properties
        @params targets: list of attack_types to train on; 'other.' means
                         any other type of attack; len(targets) >= 2
                         for binary classification
        @params layers: list of dicts each corresponding to a hidden layer; 
                        must len(layers) >= 1
        @params model_path: path file to a keras model
        """
        self.inputs = sorted(inputs)
        if targets:
            self.targets = sorted(targets)
        else:
            self.targets = sorted(kddcup_targets)
        self.layers = layers
        self.model_path = model_path
        self.loss = -1
        self.accuracy = -2

    def __getitem__(self, *args):
        if isinstance(*args, str):
            return self.__dict__[str(*args)]
        keys = list(*args)
        return [self.__dict__[key] for key in keys]

    def train(self, data, batch_size=128, epochs=5, verbose=1):
        model = Model()  # emply model with kmodel==None
        for d in data:
            model = Model(d[self.inputs][self.targets],
                          layers=self.layers, kmodel=model.kmodel)
            if model.output_dim == len(self.targets):
                print("Training..")
                model.train(batch_size=batch_size,
                            epochs=epochs, verbose=verbose)
            else:
                print("Skipping..")
        self.model = model
        return self

    def test(self, data):
        print("Testing..")
        l_a = [self.model.test(d[self.inputs][self.targets])[
            'loss', 'accuracy'] for d in data]
        l_a = np.array(l_a)
        loss, acc = l_a[:, 0].tolist(), l_a[:, 1].tolist()
        if len(loss) == 1 and len(acc) == 1:
            loss = loss[0]
            acc = acc[0]
        self.loss = loss
        self.accuracy = acc
        return self

    def save(self, path=None):
        print('Saving..')
        loss, acc = round(self.loss, 4), round(100*self.accuracy, 2)
        to_file = '-vs-'.join(self.targets)+'model-acc-{}.json'.format(acc)
        if path:
            self.model_path = os.path.join(path, to_file)
        data = {"inputs": self.inputs, "targets": self.targets,
                "k_model": self.model.kmodel.get_config()}
        with open(self.model_path, 'w') as f:
            json.dump(data, f, indent=2)
        return self

    def load(self, path=None):
        if path:
            self.model_path = path
        print('Loading..')
        with open(self.model_path) as f:
            data = json.load(f)
        self.inputs, self.targets = data["inputs"], data["targets"]
        keras_model = Sequential.from_config(data["k_model"])
        keras_model.compile(loss='binary_crossentropy',
                            optimizer='adam', metrics=['accuracy'])
        self.model = Model(kmodel=keras_model)
        return self

    def print(self):
        all_inputs, all_targets = '', ''
        if not self.inputs:
            all_inputs = '(all inputs)'
        if not self.targets:
            all_targets = '(all targets)'
        print("inputs: ", self.inputs, all_inputs)
        print("targets: ", self.targets, all_targets)
        print("layers: ", self.layers)
        print("loss: ", self.loss)
        print("accuracy: ", self.accuracy)
        return self
