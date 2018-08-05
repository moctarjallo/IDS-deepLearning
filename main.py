from data import KddCupData
from model import KddCupModel

training_data = KddCupData(filename='data/kddcup.data_10_percent_corrected', batch_size=100000)
testing_data = KddCupData(filename='data/corrected')

model = KddCupModel(layers=[{'neurons': 8, 'activation': 'relu'}],
                    output_shape=(2,))

if __name__ == '__main__':
    model.train(training_data)
    print(model.test(testing_data))

