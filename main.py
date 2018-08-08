from data import KddCupData
from model import KddCupModel

inputs = []
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']


if __name__ == '__main__':
    loss, acc = KddCupModel(KddCupData(filename='data/kddcup.data_10_percent_corrected', nrows=100000))\
                    .train(inputs=inputs, targets=targets, layers=layers, epochs=10)\
                    .test(KddCupData(filename='data/corrected', nrows=10000), inputs=inputs, targets=targets)\
                    ['loss', 'accuracy']
    print(round(loss, 4), round(100*acc, 2))
