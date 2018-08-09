from core import KddCupData, KddCupModel

train_datafile = 'data/kddcup.data_10_percent_corrected'
test_datafile = 'data/corrected'

inputs = []
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']


if __name__ == '__main__':
    # loss, acc = KddCupModel(inputs=inputs, targets=targets, layers=layers)\
    #                 .train(KddCupData(filename=train_datafile, nrows=50000), epochs=3)\
    #                 .test(KddCupData(filename=test_datafile, nrows=10000))\
    #                 .save('data/ckpts')\
    #                 ['loss', 'accuracy']

    loss, acc = KddCupModel(model_path='data/ckpts/kddcupmodel-acc97.33.hkl')\
                    .test(KddCupData(filename=test_datafile, nrows=10000))\
                    ['loss', 'accuracy']
    print(loss)
    print(acc)
    # print(round(loss, 4), round(100*acc, 2))
