from data import KddCupData
from model import KddCupModel

training_data = KddCupData(filename='data/kddcup.data_10_percent_corrected')
testing_data = KddCupData(filename='data/corrected')

X_train = training_data.X
Y_train = training_data.Y

X_test = testing_data.X
Y_test = testing_data.Y

model = KddCupModel(input_shape=(X_train.shape[1],),
                    layers=[{'neurons': 8, 'activation': 'relu'}],
                    output_shape=(2,))

if __name__ == '__main__':
    model.train(X=X_train, Y=Y_train)

