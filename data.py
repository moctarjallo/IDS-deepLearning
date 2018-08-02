import pandas as pd
import numpy as np

from sklearn import preprocessing

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
        #normal = pd.DataFrame(self.normal)
        #return self.data[~self.data.isin(normal).all(1)].values.tolist()
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



class KddCupData(object):
    def __init__(self, dataframe=None, filename='./data/kddcup.data_10_percent_corrected', nrows=None, batch_size=128):
        self.batch_size = batch_size
        self.iter = pd.read_csv(filename, names=self.__names, iterator=True, nrows=nrows)
        if dataframe is not None:
            self.current = dataframe
        else:
            self.current = self.iter.get_chunk(self.batch_size)
        self.__set_types('object', 'category')

    @property
    def __names(self):
        with open('data/kddcup.names.txt') as names_file:
            lines = names_file.readlines()[1:]
            names = [lines[i].split(':')[0] for i in range(len(lines))]
        names.append('attack_type')
        return names

    def __get_columns_by_type(self, typ):
        return list(self.current.select_dtypes(include=[typ]).columns)

    def __set_types(self, from_type, to_type):
        columns = self.__get_columns_by_type(from_type)
        if columns:
            for properti in columns:
                self.current[properti] = self.current[properti].astype(to_type)

    def __iter__(self):
        return self

    def __next__(self):
        current = self.iter.get_chunk(self.batch_size)
        return KddCupData(dataframe=current)

    @property
    def properties(self):
        return list(self.current)

    def head(self):
        return self.current.head()

    def __getitem__(self, key):
        if key in self.attack_types:
            return self.current[self.current['attack_type'] == key]
        else:
            return self.current[key]

    @property
    def attack_types(self):
        return list(self.current['attack_type'])

    @property
    def numerized(self):
        """Transform categorical types to numerical"""
        categories = self.__get_columns_by_type('category')[:-1] # not include the attack_type
        copy = self.current.copy()
        for cat in categories:
            copy[cat] = copy[cat].cat.codes
        return copy

    @property
    def normalized(self):
        """Normalize data between 0 and 1"""
        copy = self.numerized
        for properti in self.properties[:-1]:
            x = copy[properti].values.reshape(-1, 1) #returns a numpy array
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            copy[properti] = pd.DataFrame(x_scaled)        
        return copy

    def binarized(self, target_attack_type='normal.'):
        """Express the output as a binary vector with respect to the
        specific attack type"""
        copy = self.normalized
        attacks = copy['attack_type'].tolist()
        attacks = [(1, 0) if attacks[i] == target_attack_type else (0, 1) for i in range(len(attacks))]
        copy['attack_type'] = pd.Series(attacks)
        return copy




if __name__=='__main__':
    # print(data.get(properties=['protocol_type', 'service', 'attack_type'], lines=5))
    # print(data.normal.get(['service'], 5))
    # print(data.malicious.get(['service', 'attack_type'], 5))
    data = KddCupData()
    data.set_objects_to_categorical()

