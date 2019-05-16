"""
This script returns a pandas dataframe containing city_id -> all items which fall in similar_cities of city_id
"""

import pandas as pd
from collections import defaultdict
from tqdm import tqdm


class CityItems:

    """
    Create a data frame containing city -> items based on similar cities model.
    """
    def __init__(self, similar_city_docs_path, item_docs_df, item_data_df):

        """
        Initialises the class with following arguments:

        :param similar_city_docs_path: s3 path for data frame containing similar city docs
        :param item_docs_df: data frame containing similar item docs
        :param item_data_df: data frame contains items data
        """

        self.similar_city_docs = similar_city_docs_path
        self.item_docs = item_docs_df
        self.item_data = item_data_df

    def make_docs(self):

        # self.similar_city_docs = self.s3.read_parquet(self.similar_city_docs)

        # pandas can directly read from s3
        self.similar_city_docs = pd.read_csv(self.similar_city_docs)

        self.item_docs['item_id'] = self.item_docs['item_id'].astype(str)
        self.item_data['id'] = self.item_data['id'].astype(str)

        ads = self.item_data[self.item_data['id'].isin(self.item_docs['item_id'].tolist())]
        ads = ads[['id', 'city_id', 'category_id']]
        ads['city_cat'] = ads['city_id'].astype(str) + '_' + ads['category_id'].astype(str)
        city_cat_items = ads.groupby('city_cat')['id'].apply(list).to_dict()
        # self.similar_city_docs[10] = self.similar_city_docs['index'] - line causing bug

        # add a new column to random name to add at the end
        self.similar_city_docs["new_col"] = self.similar_city_docs['index']

        temp_docs = self.similar_city_docs.set_index('index').iloc[:, ::2].T.to_dict(orient='list')

        # keep only those cities_cat which are available for item_docs
        # because there are some combination of city_cat which arent available
        # because item docs has items for last 3 months
        temp_docs_2 = {k: v for k, v in temp_docs.items() if k in city_cat_items}

        # remove locations with cat 378, 801, 379, 363, 388, 367
        temp_docs_2 = {k: v for k, v in temp_docs_2.items() if
                       k.split('_')[1] not in ["378", "801", "379", "363", "388", "367"]}

        temp_docs_2 = {k: [x for x in v if str(x) != 'nan'] for k, v in temp_docs_2.items()}

        city_items = defaultdict(list)

        for k, v in tqdm(temp_docs_2.items()):
            j1 = [city_cat_items[j] for j in v if j in city_cat_items]
            city_items[k].extend(j1)
            del j1

        return city_items

    def transform_docs(self):
        city_items = self.make_docs()
        city_items = pd.DataFrame.from_dict(city_items, orient='index')
        city_items = city_items.reset_index()
        city_items = (city_items
                      .set_index('index')
                      .stack()
                      .reset_index(level=0)
                      .groupby('index')[0].apply(pd.np.concatenate)
                      .apply(lambda x: ','.join(x))
                      .reset_index()
                      .rename(columns={'index': 'city_cat', 0: 'items'}))

        return city_items



