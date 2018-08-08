from core import KddCupData, KddCupModel

train_datafile = 'data/kddcup.data_10_percent_corrected'
test_datafile = 'data/corrected'

inputs = []
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']


if __name__ == '__main__':
    loss, acc = KddCupModel(KddCupData(filename=train_datafile, nrows=50000, batch=10000))\
                    .train(inputs=inputs, targets=targets, layers=layers, epochs=3)\
                    .test(KddCupData(filename=test_datafile, nrows=10000, batch=4000), inputs=inputs, targets=targets)\
                    ['loss', 'accuracy']
    print(loss)
    print(acc)
    # print(round(loss, 4), round(100*acc, 2))
