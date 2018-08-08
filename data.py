import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

from sklearn import preprocessing

class Data:
    def __init__(self, dataframe):
        self.current = dataframe
        self.__set_types('object', 'category')

        self.inputs = None #self.properties[:-1]
        self.targets = None #list(set(self.attack_types))

    def __get_columns_by_type(self, typ):
        return list(self.current.select_dtypes(include=[typ]).columns)

    def __set_types(self, from_type='object', to_type='category', column=None):
        if column:
            columns = [column]
        else:
            columns = self.__get_columns_by_type(from_type)
        if columns:
            for properti in columns:
                self.current[properti] = self.current[properti].astype(to_type)

    @property
    def properties(self):
        return list(self.current)

    @property
    def attack_types(self):
        return list(self.current['attack_type'])

    @property
    def shape(self):
        return self.current.shape

    def head(self, n=5):
        return self.current.head(n)

    def __getitem__(self, keys):
        # keys is a list containing properties and/or attack types
        attack_keys = list(set(self.attack_types).intersection(set(keys)))
        property_keys = list(set(self.properties).intersection(set(keys)))

        self.inputs = property_keys
        self.targets = attack_keys

        if not property_keys: # if a property is not provided, consider all properties
            property_keys = self.properties
        else:
            property_keys += ['attack_type']
        if not attack_keys: # if an attack_key is not provided, consider all of them
            return Data(self.current[property_keys])
        elif 'other.' in keys: # if key word 'other' is provided, consider it as an attack type that replaces all other attack types that were not provided
                updated = self.current['attack_type'].tolist()
                for i in range(len(updated)):
                    if updated[i] not in attack_keys:
                        updated[i] = 'other.'
                self.current['attack_type'] = updated
                attack_keys += ['other.']
        c = False # make a selection condition
        for attack_key in attack_keys:
            c = c | (self.current['attack_type'] == attack_key)
   
        return Data(self.current[property_keys][c])


    @property
    def numerized(self):
        """Transform categorical types to numerical"""
        categories = self.__get_columns_by_type('category')[:-1] # not include the attack_type
        if categories:
            copy = self.current.copy()
            for cat in categories:
                copy[cat] = copy[cat].cat.codes
            return Data(copy)
        else:
            return Data(self.current)

    @property
    def normalized(self):
        """Normalize data between 0 and 1"""
        df = self.numerized.XY
        df_values = df[self.properties[:-1]].values
        norm = normalize(df_values, axis=0, copy=False)
        df[self.properties[:-1]] = norm
        return Data(df)

    @property
    def binarized(self):
        """Express the output as a binary vector with respect to the
        specific target attack. Default is 'normal.' """
        copy = self.normalized.XY
        bins = copy['attack_type'].str.get_dummies().values
        copy['attack_type'] = [tuple(b) for b in bins]
        return Data(copy) 

    @property
    def XY(self):
        """Return the whole dataframe"""
        return self.current

    @property
    def X(self):
        """Return inputs"""
        return self.XY[self.properties[:-1]]

    @property
    def Y(self):
        """Return the output"""
        return np.array(self.XY[self.properties[-1]].tolist())

class KddCupData(object):
    def __init__(self, filename='./data/kddcup.data_10_percent_corrected', nrows=None, batch=10000):
        if nrows and nrows < batch:
            nrows = batch
        self.batch = batch
        self.data = pd.read_csv(filename, names=self.__names, nrows=nrows, iterator=True)
    
    @property
    def __names(self):
        with open('data/kddcup.names.txt') as names_file:
            lines = names_file.readlines()[1:]
            names = [lines[i].split(':')[0] for i in range(len(lines))]
        names.append('attack_type')
        return names

    @property
    def properties(self):
        return self.__names

    def __iter__(self):
        return self

    def __next__(self):
        current = self.data.get_chunk(self.batch)
        self.shape = current.shape
        return Data(current)


         



if __name__=='__main__':
    data = KddCupData()


