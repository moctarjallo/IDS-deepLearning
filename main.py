from data import KddCupData
from model import KddCupModel, Model

training_data = KddCupData(filename='data/kddcup.data_10_percent_corrected', batch_size=10000)
testing_data = KddCupData(filename='data/corrected')

inputs = ['dst_bytes', 'src_bytes', 'service']
layers = [{'neurons': 8, 'activation': 'relu'},
          {'neurons': 4, 'activation': 'relu'}]
targets = ['normal.', 'other.']

# mdl = KddCupModel(inputs=inputs, targets=targets, layers=layers)
mdl = Model(next(training_data)[inputs][targets], layers=layers)

if __name__ == '__main__':
    mdl = None
    for d in training_data:
        if mdl:
            # Transfer the previously learned weights for the next training
            mdl = Model(d[inputs][targets], model=mdl.model)
        else:
            mdl = Model(d[inputs][targets], layers=layers)
        if mdl.output_dim == len(targets):
            mdl.train()
        else:
            print("Skipping..")
    print(mdl.test(next(testing_data)[inputs][targets]))

