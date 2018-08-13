from core.data import KddCupData
from core.model import KddCupModel

from kddcup.core.constants import training_file, testing_file

inputs = []
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']


if __name__ == '__main__':
    loss, acc = KddCupModel(inputs=inputs, targets=targets)\
                    .train(KddCupData(training_file, nrows=100000), epochs=3)\
                    .test(KddCupData(testing_file, nrows=10000))\
                    .save('kddcup/data/ckpts')\
                    .print()\
                    ['loss', 'accuracy']

    # loss, acc = KddCupModel(model_path='data/ckpts/model-acc97.33.kdd')\
    #                 .test(KddCupData(filename=test_datafile, nrows=10000))\
    #                 ['loss', 'accuracy']
    print(loss)
    print(acc)
    # print(round(loss, 4), round(100*acc, 2))
