import pandas as pd
import numpy as np
import hypertools as hyp


class Data:
    def __init__(self, datafile='data/kddcup.data_10_percent_corrected', datalist=None, nrows=None):
        if datalist:
            self.data = pd.Dataframe(datalist)
        elif datafile:
            self.data = pd.read_csv(datafile, names=self.properties, nrows=nrows)
            self.set_types(['protocol_type', 'service', 'flag', 'attack_type'], 4*['category'])

    """Size of this dataset"""
    @property
    def size(self):
        return len(self.data)

    """List of packet properties read from a file"""
    @property
    def properties(self):
        with open('data/kddcup.names.txt') as names_file:
            lines = names_file.readlines()[1:]
            names = [lines[i].split(':')[0] for i in range(len(lines))]
        names.append('attack_type')
        return names

    """Return a Data object containing specified list of properties"""
    def get(self, properties=['all'], lines='all', attack_type=None):
        if attack_type:
            data = self.data[self.data['attack_type'] == attack_type]
        else:
            data = self.data
        if properties == ['all'] and lines == 'all':
            pass
        elif properties == ['all']:
            try:
                data = data[:lines]
            except ValueError:
                print("wrong argument lines")
        elif lines == 'all':
            try:
                data = data[properties]
            except ValueError:
                print("wrong argument properties: must be list of existing properties")
        else:
            try:
                data = data[properties][:lines]
            except ValueError:
                print("wrong argument lines")
        return data.values.tolist()


    """Update a specified property using a list of function updaters"""
    def update(self, properties, updaters):
        for property, updater in zip(properties, updaters):
            self.data[property] = self.data[property].apply(updater)

    """Get the type of each property listed in properties"""
    def get_types(self, properties):
        return self.data[properties].dtypes

    """Set the type of listed properties to the listed types"""
    def set_types(self, properties, types):
        for properti, typ in zip(properties, types):
            self.data[properti] = self.data[properti].astype(typ)
            
    def set_cat_to_num(self, properties):
        for properti in properties:
            self.data[properti] = self.data[properti].cat.codes

    """Return a Data object containing malicious packets"""
    @property
    def malicious(self):
#         normal = pd.DataFrame(self.normal)
#         return self.data[~self.data.isin(normal).all(1)].values.tolist()
        return self.data[self.data['attack_type'] != self.data.iloc[1, -1]].values.tolist()


    """Return a Data object containing normal packets"""
    @property
    def normal(self):
        return self.data[self.data['attack_type'] == self.data.iloc[1, -1]].values.tolist()


    @property
    def attack_categories(self):
        return self.categories('attack_type')

    def plot(self, properties=['all'], labels=None, ndims=None, nsamples=None):
        hyp.plot(self.get(properties, lines=nsamples), labels=labels, ndims=ndims)

    def get_categories(self, properti):
        assert type(properti) == str, "properti input must be a string"
        return list(self.data[properti].dtype.categories)


if __name__=='__main__':
    data = Data()
    # print(data.get(properties=['protocol_type', 'service', 'attack_type'], lines=5))
    # print(data.normal.get(['service'], 5))
    print(data.malicious.get(['service', 'attack_type'], 5))
