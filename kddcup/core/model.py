from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

import numpy as np

import os


class Model(object):
    def __init__(self, data=None, layers=[], model=None, model_path=None):
        """

        data: object of type data.Data
        layers: a list of dicts, each representing a layer
        model: object of type of keras.models.Sequential
        model_path: filepath where a keras model is saved
        """
        if data:
            self.data = data.binarized
        if model:
            self.model = model.model
        elif model_path:
            self = self.load(model_path)
        else:
            self.model = self.__build_model(layers)
        self.inputs = []
        self.targets = []
        self.loss = -1
        self.accuracy = -2


    def __getitem__(self, *args):
        if isinstance(*args, str):
            return self.__dict__[str(*args)]
        keys = list(*args)
        return [self.__dict__[key] for key in keys]

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

    def train(self, batch_size=128, epochs=10, verbose=1):
        self.model.fit(self.data.X, self.data.Y, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return self

    def test(self, data=None):
        if data:
            data = data.binarized
        else:
            data = self.data
        self.loss, self.accuracy = self.model.evaluate(data.X, data.Y)
        return self

    def save(self, path):
        loss, acc = round(self.loss, 4), round(100*self.accuracy, 2)
        to_file = '-vs-'.join(self.targets)+'model-acc-{}.h5'.format(acc)
        self.model.save(os.path.join(path, to_file))
        return self

    def load(self, path):
        keras_model = load_model(path)
        return Model(model=keras_model)

    
class KddCupModel(object):
    def __init__(self, inputs=[], targets=[], layers=[{'neurons': 1, 'activation': 'relu'}], 
                       model=None, model_path=None):
        """Model of KddCup data

        @params inputs: list of input properties to consider; default to [] 
                        meaning all properties
        @params targets: list of attack_types to train on; 'other.' means
                         any other type of attack; len(targets) >= 2
                         for binary classification
        @params layers: list of dicts each corresponding to a hidden layer; 
                        must len(layers) >= 1
        @params model: object of type model.Model
        """

        self.layers = layers
        if model_path:
            self.model = Model(model_path=model_path)
            self.inputs = self.model.inputs
            self.targets = self.model.targets
        elif model:
            model.inputs = inputs
            model.targets = targets
            self.model = model
        else:
            self.model = None
            self.inputs = sorted(inputs)
            self.targets = sorted(targets)
        self.loss = -1
        self.accuracy = -2

    def __getitem__(self, *args):
        if isinstance(*args, str):
            return self.__dict__[str(*args)]
        keys = list(*args)
        return [self.__dict__[key] for key in keys]
    
    def train(self, data, batch_size=128, epochs=5, verbose=1):
        for d in data:
            self.model = Model(d[self.inputs][self.targets], layers=self.layers, model=self.model)
            if self.model.output_dim == len(self.targets):
                print("Training..")
                self.model.train(batch_size=batch_size, epochs=epochs, verbose=verbose)
            else:
                print("Skipping..")
            self.model.inputs = self.inputs
            self.model.targets = self.targets
        return self

    def test(self, data):
        print("Testing..")
        l_a = [self.model.test(d[self.inputs][self.targets])['loss', 'accuracy'] for d in data]
        l_a = np.array(l_a)
        loss, acc = l_a[:, 0].tolist(), l_a[:, 1].tolist()
        if len(loss) == 1 and len(acc) == 1:
            loss = loss[0]
            acc = acc[0]
        self.loss = loss
        self.accuracy = acc
        return self

    def save(self, path):
        print("Saving..")
        self.model.save(path)
        return self

    def load(self, path):
        print("Loading..")
        self.model.load(path)
        return self

    def print(self):
        print("inputs: ", self.inputs)
        print("targets: ", self.targets)
        print("loss: ", self.loss)
        print("accuracy: ", self.accuracy)
        return self