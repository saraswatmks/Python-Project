import numpy as np
import pandas as pd

## TODO : Refactor it to calculate item similarity
class Embedding(object):
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embeddings_size = len(embeddings)
        self._embeddings_array = self.embeddings.values.astype('float32').copy()
        self._search_string = self.embeddings.index

    def _to_search_string(self, index):
        return self._search_string[index]

    def _to_search_index(self, query):
        return self._search_string.get_loc(query)

    def related_from_index(self, index, related=5):
        a = self._embeddings_array.dot(self._embeddings_array[index, :])
        ind = np.argpartition(a, -(related + 1))[-(related + 1):]
        similar_pairs = sorted(list(zip(self._to_search_string(ind), a[ind])), key=lambda x: x[1], reverse=True)[1:]
        return similar_pairs

    def related(self, query, **kwargs):
        index = self._to_search_index(query)
        return self.related_from_index(index, **kwargs)

    def related_as_df(self, query, **kwargs):
        return pd.DataFrame([{'related_query': k, 'score': v} for k, v in self.related(query, **kwargs)])

    def score(self, query1, query2):
        index1 = self._to_search_index(query1)
        index2 = self._to_search_index(query2)
        return self._embeddings_array[index1, :].dot(self._embeddings_array[index2, :])
