"""
This script returns a pandas data frame containing city_id -> titles of all ads which fall in cities similar to city_id
This script is dependent on output of city_items.py
"""

from tqdm import tqdm
from vision.train.city_items import CityItems


class CityTitles(CityItems):
    def __init__(self, similar_city_docs_path, item_docs_df, item_params_df, item_data_df):

        """

        Initialises the class with the following arguments:

        :param similar_city_docs_path: s3 path for data frame containing similar city docs
        :param item_docs_df: data frame containing similar item docs
        :param item_params_df: data frame containing items data
        """
        self.similar_city_docs = similar_city_docs_path
        self.item_docs = item_docs_df
        self.item_params = item_params_df
        super().__init__(similar_city_docs_path=similar_city_docs_path, item_docs_df=item_docs_df, item_data_df=item_data_df)
        self.city_item_docs = self.transform_docs()

    def generate_docs(self):
        self.item_params['item_id'] = self.item_params['item_id'].astype(str)
        item_title_dict = self.item_params.set_index('item_id')['title'].to_dict()
        self.city_item_docs['items'] = self.city_item_docs['items'].str.split(',')
        self.city_item_docs['titles'] = [[item_title_dict[y] for y in x if y in item_title_dict] for x in
                                tqdm(self.city_item_docs['items'])]

        # self.city_item_docs['titles'] = self.city_item_docs['titles'].str.join(',')
        # self.city_item_docs['items'] = self.city_item_docs['items'].str.join(',')
        return self.city_item_docs




