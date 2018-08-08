from data import KddCupData
from model import KddCupModel, Model

training_data = KddCupData(filename='data/kddcup.data_10_percent_corrected', nrows=10000, batch=10000)
testing_data = KddCupData(filename='data/corrected', batch=50000)

inputs = ['dst_bytes', 'src_bytes', 'service']
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']


if __name__ == '__main__':
    loss, acc = KddCupModel(training_data)\
                    .train(inputs, targets, layers, epochs=1)\
                    .test(testing_data)['loss', 'accuracy']
    # print(round(loss, 4), round(100*acc, 2))
    print(loss, acc)

