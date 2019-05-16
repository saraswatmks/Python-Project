"""
This script returns a pandas data frame containing item formation. The information such as
'item_id', 'title', 'category_id', 'lat', 'lon', 'cat_attr' are available. Here the title field is generated
such that for each item, we have titles of all its similar item. This file will be used to create db.

"""

import pandas as pd
import re
from tqdm import tqdm
import os
from vision.vision import VisionLogger


class ItemParams(VisionLogger):

    def __init__(self, item_data, item_docs, is_local=None):

        """
        initialises the class with following arguments:

        :param item_data: a data frame containing item attributes information like category_id, title, description etc
        :param item_docs: a data frame containing item -> similar items
        """
        super().__init__()
        self.item_data = item_data
        self.item_docs = item_docs
        self.path = "STOPWORDS.txt" if not is_local else \
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "docs/STOPWORDS.txt")
        self.stopwords = open(self.path, 'r').read().splitlines()

    def generate_docs(self):
        self.item_data['id'] = self.item_data['id'].astype(str)
        self.item_docs['related_item_100'] = self.item_docs['item_id'] # use any random column name

        df = (self.item_docs
              .set_index('item_id')
              .iloc[:, ::2]
              .stack()
              .reset_index()
              .drop('level_1', axis=1)
              .rename(columns={'index': 'item_id', 0: 'related_item'}))

        df['related_item'] = df['related_item'].astype(str)
        df['item_id'] = df['item_id'].astype(str)
        df = pd.merge(df, self.item_data[['id', 'title']], how="left", left_on='related_item', right_on='id')
        df = df.reset_index(drop=True)
        df['title'] = df['title'].astype(str)
        df['title'] = [self.clean_title(x) for x in tqdm(df['title'])]
        df = df.groupby('item_id')['title'].apply(list).reset_index()
        cols = self.item_data.columns.difference(['title', 'description', 'timestamp']).tolist()

        df2 = df.merge(self.item_data[cols], how="left", left_on="item_id", right_on="id")
        return df2

    def clean_title(self, query):

        # remove punc marks
        query = re.sub(r'[^a-zA-Z0-9.\s ]+', ' ', query)
        query = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', query)
        query = re.sub(r'\s\s+', ' ', query)

        return ' '.join([x for x in query.lower().strip().split() if x not in self.stopwords])

    def myfunc(self, category, params):

        if category == "379":  # for motorcycle
            return params['make']
        elif category == "378":  # cars
            return params['model']
        elif category == "801":  # Mobile
            return params['brand']
        elif category in ["363", "388", "367"]:  # houses
            return params['bedrooms']
        else:
            return None

    def transform_docs(self):
        df = self.generate_docs()
        df['category_id'] = df['category_id'].astype(str)
        df['attrs'] = df.apply(lambda x: self.myfunc(x['category_id'], x), axis=1)
        df['attrs'] = df['attrs'].astype(str)

        df['cat_attr'] = df.apply(lambda x: ','.join([x['category_id'], x['attrs']]), axis=1)
        df['title'] = [' '.join(x) for x in df['title']]
        df.rename(columns={'map_lat': 'lat', 'map_lon': 'lon'}, inplace=True)
        return df[['item_id', 'title', 'category_id', 'lat', 'lon', 'cat_attr']]

