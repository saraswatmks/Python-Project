import pandas as pd
import numpy as np

class CreateData:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def get_data(self):
        return self.a, self.b

    def get_df(self):
        mk = pd.DataFrame({'col1': [self.a, self.b],
                           'col2': [self.b, self.a]})
        return mk

    def get_array(self):
        mk = np.random.randn(5)
        return mk
