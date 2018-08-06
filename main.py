from data import KddCupData
from model import KddCupModel, Model

training_data = KddCupData(filename='data/kddcup.data_10_percent_corrected', batch_size=10000)
testing_data = KddCupData(filename='data/corrected')

inputs = []
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']

# mdl = KddCupModel(inputs=inputs, targets=targets, layers=layers)
mdl = Model(next(training_data)[['normal.', 'other.']], layers=layers)

if __name__ == '__main__':
    mdl.train()
    print(mdl.test(next(testing_data)[['normal.', 'other.']]))

