"""kddcup program

Usage:
    kddcup train [--file=<filepath> --nrows=<int> --batch=<int> --batch_train=<int> --epochs=<int> --verbose=<int> --save_path=<filepath>] 
    kddcup test [--model=<modelpath> --file=<filepath> --nrows=<int> --batch=<int>  --verbose=<int>]
    kddcup predict [--packet=<csvfile>]
    kddcup plot
    kddcup -h | --help

Options:
    --file=<filepath>         data file to be trained/tested on
    --nrows=<int>             number of rows to read from FILE [default: 10000]
    --batch=<int>             iteration size over data being read from FILE [default: 10000]
    --batch_train=<int>       training batch [default: 128]
    --epochs=<int>            epochs [default: 5]
    --verbose=<int>           show the training [default: 1]
    --model=<modelpath>       path to a saved model
    -h --help                 Show this screen
"""

from kddcup.core.data import KddCupData
from kddcup.core.model import KddCupModel

from kddcup.evolution import Population

from plot import app

import json
import os
import pathlib

from docopt import docopt


HOME = str(pathlib.Path.home())


class KddCup(object):
    def __init__(self, config_file="./kddcup/config.json"):
        with open(config_file) as f:
            config = json.load(f)
        self.config = config
        self.home = str(pathlib.Path.home())

    """Train a model on input data and return loss and accuracy"""

    def train(self):
        train_data = KddCupData(filename=os.path.join(self.home, self.config["train"]["file"]),
                                nrows=self.config["train"]["rows"],
                                batch=self.config["train"]["batch_iter"])
        model = KddCupModel(inputs=self.config["inputs"],
                            targets=self.config["targets"],
                            layers=self.config["layers"])
        model = model.train(data=train_data,
                            batch_size=self.config["train"]["batch_train"],
                            epochs=self.config["train"]["epochs"],
                            verbose=self.config["train"]["verbose"])
        if self.config["train"]["save_model"]:
            model.save(path=os.path.join(
                self.home, self.config["train"]["save_model"]))
        return model['model_path']

    def test(self):
        data = KddCupData(filename=os.path.join(self.home, self.config["test"]["file"]),
                          nrows=self.config["test"]["rows"],
                          batch=self.config["test"]["batch_iter"])
        model = KddCupModel(
            model_path=os.path.join(self.home, self.config["test"]["model_path"])).load()
        return model.test(data)['loss', 'accuracy']

    def evolve(self):
        pop = Population(space=self.config["kddcup_properties"][:-1],
                         targets=self.config["targets"],
                         brain=self.config["layers"])
        return pop.evolve(train_env=os.path.join(self.home, self.config["train"]["file"]),
                          test_env=os.path.join(
                              self.home, self.config["test"]["file"]),
                          train_surface=self.config["train"]["rows"],
                          train_hops=self.config["train"]["batch_iter"],
                          train_iterations=self.config["train"]["epochs"],
                          train_verbose=self.config["train"]["verbose"],
                          test_surface=self.config["test"]["rows"],
                          NGEN=self.config["evolve"]["generations"],
                          MUTPB=self.config["evolve"]["mutation_prob"],
                          CXPB=self.config["evolve"]["crossover_prob"]
                          )

    def predict(self, packet):
        pass


# def train(file, nrows=10000, batch=None, batch_train=128, epochs=10, verbose=1, save_path='.kddcup/ckpts'):
#     train_data = KddCupData(filename=os.path.join(
#         HOME, file), nrows=nrows, batch=batch)
#     model = KddCupModel(targets=['normal.', 'other.'])
#     return model.train(data=train_data, batch_size=batch_train, epochs=epochs, verbose=verbose)\
#         .test(data=KddCupData(filename=os.path.join(HOME, file), nrows=nrows, batch=batch))\
#         .save(path=os.path.join(HOME, save_path))['model_path']


def cli():
    program = KddCup()
    args = docopt(__doc__)
    if args['train']:
        if args['--file']:
            program.config["train"]["file"] = args['file']
        if args['--nrows']:
            program.config["train"]["rows"] = int(args['--nrows'])
        if args['--batch']:
            program.config["train"]["batch_iter"] = int(args['--batch'])
        if args['--batch_train']:
            program.config["train"]["batch_train"] = int(args['--batch_train'])
        if args['--epochs']:
            program.config["train"]["epochs"] = int(args['--epochs'])
        if args['--verbose']:
            program.config["train"]["verbose"] = int(args['--verbose'])
        if args['--save_path']:
            program.config["train"]["save_model"] = args['--save_path']
        model = program.train()
        print(model)
    if args['test']:
        print(args)
        if args['--model']:
            program.config["test"]["model_path"] = args['--model']
        if args['--file']:
            program.config["test"]["file"] = args['file']
        if args['--nrows']:
            program.config["test"]["rows"] = int(args['--nrows'])
        if args['--batch']:
            program.config["test"]["batch_iter"] = int(args['--batch'])
        if args['--verbose']:
            program.config["test"]["verbose"] = int(args['--verbose'])
        loss, acc = program.test()
        print(loss, acc)
    if args['plot']:
        app.run_server(debug=True)


if __name__ == '__main__':
    cli()
