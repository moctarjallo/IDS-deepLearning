from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint


class KddCupModel(object):
    def __init__(self, input_shape=None, layers=[], output_shape=2):
        self.callbacks = [TensorBoard(log_dir='../logs/tensorboard/15'), ModelCheckpoint(filepath='../logs/models/model-last.h5')]
        self.model = self.__build_model(input_shape=input_shape, layers=layers, output_shape=output_shape)


    def __build_model(self, input_shape=None, layers=[], output_shape=None):
        # layers = [{'neurons': neurons, 'activation': activation, ...}, 
                  # {'neurons': neurons, 'activation': activation, ...},
                  #  ... ]
        # input_shape is a tuple
        model = Sequential()
        model.add(Dense(layers[0]['neurons'], input_shape=input_shape, activation=layers[0]['activation']))
        for layer in layers[1:]:
            model.add(Dense, layer['neurons'], activation=layer['activation'])
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X, Y, batch_size=128, epochs=10):
        self.model.fit(batch_size=batch_size, epochs=epochs, x=X, y=Y)

    def test(self, X, Y):
        loss, acc = self.model.evaluate(x=X, y=Y)
        return round(loss, 4), round(acc, 4)