from kddcup.core.data import KddCupData
from kddcup.core.model import KddCupModel

from kddcup.core.constants import training_file, testing_file

inputs = []
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']


if __name__ == '__main__':
    # loss, acc = KddCupModel(inputs=inputs, targets=targets)\
    #                 .train(KddCupData(training_file, batch=100000), epochs=15)\
    #                 .test(KddCupData(testing_file))\
    #                 .save('/home/mctrjalloh/.kddcup/ckpts')\
    #                 .print()\
    #                 ['loss', 'accuracy']

    loss, acc = KddCupModel(model_path='/home/mctrjalloh/.kddcup/ckpts/normal.-vs-other.model-acc-91.6.h5',
                            inputs=inputs, targets=targets)\
                    .test(KddCupData(filename=testing_file))\
                    .print()\
                    ['loss', 'accuracy']
    print(loss)
    print(acc)
    # print(round(loss, 4), round(100*acc, 2))
