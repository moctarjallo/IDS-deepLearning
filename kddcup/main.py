from kddcup.core.data import KddCupData
from kddcup.core.model import KddCupModel

from kddcup.evolution import Population

import json
import os
import pathlib


class KddCup(object):
    def __init__(self, config_file="./config.json"):
        with open(config_file) as f:
            config = json.load(f)
        self.config = config
        self.home = str(pathlib.Path.home())

    """Train a model on input data and return loss and accuracy"""

    def train(self):
        train_data = KddCupData(filename=os.path.join(self.home, self.config["train"]["file"]),
                                nrows=self.config["train"]["rows"],
                                batch=self.config["train"]["batch_iter"])
        test_data = KddCupData(filename=os.path.join(self.home, self.config["test"]["file"]),
                               nrows=self.config["test"]["rows"],
                               batch=self.config["test"]["batch_iter"])
        model = KddCupModel(inputs=self.config["inputs"],
                            targets=self.config["targets"],
                            layers=self.config["layers"])
        return model.train(data=train_data,
                           batch_size=self.config["train"]["batch_train"],
                           epochs=self.config["train"]["epochs"],
                           verbose=self.config["train"]["verbose"])\
            .test(data=test_data)\
            .save(path=self.config["train"]["save_model"])['model_path']

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


if __name__ == '__main__':
    # model_path = KddCup('kddcup/config.json').train()
    loss, acc = KddCup('kddcup/config.json').test()
    print(loss, acc)
    # print(model_path)
