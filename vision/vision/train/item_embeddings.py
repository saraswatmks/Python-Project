"""
This script returns item -> [(sim_item, score), (sim_item, score)]
"""

from tqdm import tqdm
import numpy as np
from vision.vision import VisionLogger
from math import radians, cos, sin, asin, sqrt


class ItemEmbedding(VisionLogger):
    cat_attr = {379: 'make', 378: 'model', 801: 'brand', 363: 'bedrooms', 388: 'bedrooms', 367: 'bedrooms'}

    def __init__(self, embedding_df, item_attributes, item_category, category_item, threshold, distance):

        super().__init__()
        self.item_to_attr = item_attributes
        self.item_to_cat = item_category
        self.threshold = threshold
        self.embeddings = {}
        self.get_embeddings(item_attributes, category_item, embedding_df)
        self.distance = distance

    def get_embeddings(self, item_to_attr, cat_to_item, embedding_df):

        for cat, items in tqdm(cat_to_item.items()):

            # self.logger.info(embedding_df.loc[embedding_df.index.isin(items)])

            if cat == 379:
                attr_temp = set(item_to_attr[item][0]['params']['make'] for item in items if
                                item_to_attr[item][0]['params']['make'] is not None)

                for attr in attr_temp:
                    items2 = [item for item in items if item_to_attr[item][0]['params']['make'] == attr]

                    if len(items2) > 0:
                        self.embeddings[(cat, attr)] = embedding_df.loc[embedding_df.index.isin(items2)]

                self.embeddings[(cat, None)] = embedding_df.loc[embedding_df.index.isin(items)]

            elif cat == 378:
                attr_temp = set(item_to_attr[item][0]['params']['model'] for item in items if
                                item_to_attr[item][0]['params']['model'] is not None)

                for attr in attr_temp:
                    items2 = [item for item in items if item_to_attr[item][0]['params']['model'] == attr]

                    if len(items2) > 0:
                        self.embeddings[(cat, attr)] = embedding_df.loc[embedding_df.index.isin(items2)]

                self.embeddings[(cat, None)] = embedding_df.loc[embedding_df.index.isin(items)]

            elif cat == 801:
                attr_temp = set(item_to_attr[item][0]['params']['brand'] for item in items if
                                item_to_attr[item][0]['params']['brand'] is not None)

                for attr in attr_temp:
                    items2 = [item for item in items if item_to_attr[item][0]['params']['brand'] == attr]
                    if len(items2) > 0:
                        self.embeddings[(cat, attr)] = embedding_df.loc[embedding_df.index.isin(items2)]

                self.embeddings[(cat, None)] = embedding_df.loc[embedding_df.index.isin(items)]

            elif cat in [363, 388, 367]:

                attr_temp = set(item_to_attr[item][0]['params']['bedrooms'] for item in items if
                                item_to_attr[item][0]['params']['bedrooms'] is not None)

                for attr in attr_temp:
                    items2 = [item for item in items if item_to_attr[item][0]['params']['bedrooms'] == attr]

                    if len(items2) > 0:
                        self.embeddings[(cat, attr)] = embedding_df.loc[embedding_df.index.isin(items2)]

                self.embeddings[(cat, None)] = embedding_df.loc[embedding_df.index.isin(items)]
            else:
                self.embeddings[(cat, None)] = embedding_df.loc[embedding_df.index.isin(items)]

    def filter_by_distance(self, query_location, items):

        items2 = [self.item_to_attr[item][0]['location'] for item in items]

        distance_check = []

        for i, x in enumerate(items2):

            dist = ItemEmbedding.haversine(float(x[1]), float(x[0]), float(query_location[1]), float(query_location[0]))
            if dist <= self.distance:
                distance_check.append((i, dist))
        distance_check = sorted(distance_check, key=lambda y: y[1])
        return set(items[i[0]] for i in distance_check)

    @staticmethod
    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))
        # Radius of earth in kilometers is 6371
        km = 6371 * c
        return km

    def related(self, query, related=10):

        # self.logger.info("this is the query: %s" % query)

        head_cat = self.item_to_cat[query]

        # self.logger.info("this is head cat: %s" % head_cat)

        attr = self.item_to_attr[query][0]['params'].get(self.cat_attr.get(head_cat, None), None)

        # self.logger.info("this is head attr: %s" % attr)

        head_location = self.item_to_attr[query][0]['location']

        # self.logger.info("this is head location")
        # self.logger.info(head_location)

        embedding = self.embeddings[(head_cat, attr)]

        # self.logger.info('these are embeddings shape')
        # self.logger.info(embedding.shape)

        # self.logger.info('this is threshold: %s' % self.threshold)

        # find dot product
        # head_item_index = embedding.index.get_loc(query)
        embeddings_array = embedding.values

        a = embeddings_array.dot(embedding.loc[query])
        ind = np.where(a >= self.threshold)
        similar_pairs = sorted(zip(embedding.index[ind], a[ind]), key=lambda x: x[1], reverse=True)[1:]

        # self.logger.info('these are similar_pairs')
        # self.logger.info(similar_pairs)

        near_items = self.filter_by_distance(head_location, embedding.index[ind].tolist())

        # self.logger.info("these are near items")
        # self.logger.info(near_items)

        # take all the items above threshold
        return [(k, v) for k, v in similar_pairs if k in near_items and k != query][:related]




