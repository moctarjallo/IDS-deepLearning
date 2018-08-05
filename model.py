from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint


class KddCupModel(object):
    def __init__(self, layers=[], output_shape=2):
        self.callbacks = [TensorBoard(log_dir='../logs/tensorboard/15'), ModelCheckpoint(filepath='../logs/models/model-last.h5')]
        self.layers = layers
        self.output_shape = output_shape
        # self.model = self.__build_model(self.input_shape)

    def __build_model(self, input_shape):
        # layers = [{'neurons': neurons, 'activation': activation, ...}, 
                  # {'neurons': neurons, 'activation': activation, ...},
                  #  ... ]
        # input_shape is a tuple
        model = Sequential()
        model.add(Dense(self.layers[0]['neurons'], input_shape=input_shape, activation=self.layers[0]['activation']))
        for layer in self.layers[1:]:
            model.add(Dense, layer['neurons'], activation=layer['activation'])
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, data, batch_size=128, epochs=10):
        # for d in data:
        #     model = self.__build_model(d.X.shape[1])
        #     model.fit(batch_size=batch_size, epochs=epochs, x=d.X, y=d.Y)
        d = next(data)['normal.', 'other.'].binarized
        self.model = self.__build_model((d.X.shape[1],))
        self.model.fit(x=d.X, y=d.Y, batch_size=batch_size, epochs=10)

    def test(self, data):
        d = next(data)['normal.', 'other.'].binarized
        loss, acc = self.model.evaluate(x=d.X, y=d.Y)
        return round(loss, 4), round(acc, 4)