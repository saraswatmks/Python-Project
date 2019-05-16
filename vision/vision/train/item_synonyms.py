"""
This script returns a pandas data frame having item -> related_item, related_score
"""


import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pygeohash as pgh
from vision.train.item_embeddings import ItemEmbedding
import time
from multiprocessing import Manager, Process
import numpy as np
from vision.train.item_params import ItemParams


class ItemSynonyms(ItemParams):
    def __init__(self,
                 item_embeddings,
                 item_attributes,
                 date_filter,  # "2016-12-01"
                 similarity_threshold=0.7,
                 distance_threshold_km=200,
                 top_related=15,
                 is_local=None):

        """
        Initialises the class with paths to item_attributes, item_embeddings file and
        with the path where the item docs will be saved.

        :param item_attributes_path: file to item embedding - output of training
        :param item_embeddings_path: file to item attributes - output of make_item_data
        :param s3_output_path: save item docs to this path: ex: s3://olx-relevance-panamera/item2vec/item_docs.csv
        """
        self.synonyms = defaultdict(list)
        self.item_attributes = item_attributes
        self.item_embeddings = item_embeddings
        self.date_filter = date_filter
        self.similarity_threshold = similarity_threshold
        self.distance_threshold_km = distance_threshold_km
        self.top_related = top_related
        self.item_to_category = None
        super().__init__(item_docs=None, item_data=None, is_local=is_local)

    @staticmethod
    def encode(loc):
        lat, lon = loc
        return pgh.encode(float(lat), float(lon), precision=8)

    @staticmethod
    def decode(loc):
        val = pgh.decode(ItemSynonyms.encode(loc))
        return [val[0], val[1]]

    def item_to_attributes_map(self, attributes):
        attributes['hashed_location'] = [self.decode((x, y)) for x, y in tqdm(zip(attributes['map_lat'], attributes['map_lon']))]
        cols_to_map = ['hashed_location', 'bedrooms', 'brand', 'brand_model', 'make', 'model']
        attributes['params'] = attributes[cols_to_map].apply(lambda x: [{'location': x['hashed_location'],
                                                                  'params': {'bedrooms': x['bedrooms'],
                                                                             'brand': x['brand'],
                                                                             'brand_model': x['brand_model'],
                                                                             'make': x['make'],
                                                                             'model': x['model']}}], axis=1)
        item_to_atr = attributes.set_index('id')['params'].to_dict()
        return item_to_atr

    def cat_to_item(self):
        ci = defaultdict(list)
        for k, v in self.item_to_category.items():
            ci[v] += [k]
        return ci

    def item_to_cat(self):
        # item_to_cat should be created after filtering the data
        self.item_to_category = self.item_attributes.set_index("id")["category_id"].to_dict()

    def filter_data(self):
        # convert id to str
        self.item_embeddings['item_id'] = self.item_embeddings['item_id'].astype(str)
        self.item_attributes["id"] = self.item_attributes["id"].astype(str)
        self.item_attributes['created_at'] = pd.to_datetime(self.item_attributes['created_at'])
        self.logger.info('this is embedding size')
        self.logger.info(self.item_embeddings.shape)
        self.logger.info(self.item_embeddings.columns)
        self.logger.info('this is attributes size')
        self.logger.info(self.item_attributes.shape)
        self.logger.info(self.item_attributes.columns)

        # take items from last 3 months
        items = self.item_embeddings[["item_id"]]

        self.logger.info("this is intersection len: %s" % len(set(items['item_id']).intersection(self.item_attributes['id'])))
        self.logger.info("this is the date for filter: %s", self.date_filter)

        items = items.merge(self.item_attributes[['id', 'created_at', 'status']], left_on='item_id', right_on='id')

        items = items[items['created_at'] >= pd.Timestamp(self.date_filter)]

        self.logger.info(f"before sold filter, we have item left: {len(items)}")

        items = items[items['status'].isin(['active', 'new'])]

        self.logger.info(f"after sold items filter, we have items left: {len(items)}")

        # filter embeddings and keep items from last 3 months
        self.item_embeddings = self.item_embeddings[self.item_embeddings["item_id"].isin(items["item_id"].tolist())].reset_index(drop=True)
        self.item_attributes = self.item_attributes[self.item_attributes["id"].isin(items["item_id"].tolist())].reset_index(drop=True)

        self.logger.info(f"after date filter, we have item for which we compute synonyms: {self.item_embeddings.shape}")

        # create item_to_cat dict now
        self.item_to_cat()

    def related_items(self):

        im = ItemEmbedding(embedding_df=self.item_embeddings.set_index("item_id")
                           , item_attributes=self.item_to_attributes_map(self.item_attributes)
                           , item_category=self.item_to_category
                           , category_item=self.cat_to_item()
                           , threshold=self.similarity_threshold
                           , distance=self.distance_threshold_km
                           )

        def wrap(d, l):

            for i in tqdm(l):
                d[i] = im.related(i, related=self.top_related)

        num_process = 6
        item_chunks = np.array_split(self.item_embeddings["item_id"].tolist(), num_process)

        self.logger.info(f"len of item chunks: {len(item_chunks)}")
        # self.logger.info(f"this is item_chunks: {item_chunks}")
        self.logger.info(f"this is chunks len: {[len(x) for x in item_chunks]}")

        with Manager() as manager:
            synonyms = manager.dict()
            start_time = time.time()
            self.logger.info(f"multiprocessing started at: {start_time}")

            processes = [Process(target=wrap, args=(synonyms, item_chunks[n])) for n in range(num_process)]

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            self.synonyms = dict(synonyms.items())
            self.logger.info(f"multiprocessing finished at: {time.time()}")
            self.logger.info(f'total time elapsed: {time.time() - start_time}')
            self.logger.info(f"this is average len of similar items: "
                             f"{round(np.mean([len(v) for k,v in self.synonyms.items()]))}")
            self.logger.info(f"this is synonyms dict: {list(self.synonyms.items())[:10]}")

    @staticmethod
    def textSimilarity(t1, t2):

        if t1 is None:
            t1 = ""

        if t2 is None:
            t2 = ""

        return len(set(t1.lower().split()).intersection(t2.lower().split()))

    def sort_by_text(self):

        self.logger.info("this is inside sort by text")
        self.synonyms = {k: v for k, v in self.synonyms.items() if len(v) > 1}
        self.logger.info("len of synonyms dict after filtering is: %s" % len(self.synonyms))
        self.synonyms = {k: [i for s in v for i in s] for k, v in self.synonyms.items()}

        self.item_attributes['title'] = self.item_attributes['title'].apply(self.clean_title)
        item_attr_dict = dict(zip(self.item_attributes['id'], self.item_attributes['title']))

        sym = {k: dict(zip(v[::2], v[1::2])) for k, v in self.synonyms.items()}
        sym = {str(i): {str(k): v for k, v in j.items() if str(k) != 'nan'} for i, j in sym.items()}

        self.logger.info("starting sorting by text")
        sym = {k: sorted(v.items(), key=lambda x: ItemSynonyms.textSimilarity(item_attr_dict.get(x[0], None),
                                                                              item_attr_dict.get(k, None)),
                         reverse=True) for k, v in sym.items()}
        sym = {k: [i for i in v if i[0] != k] for k, v in sym.items()}
        sym = {k: [i for s in v for i in s] for k, v in sym.items()}

        return sym

    def transform_docs(self):

        docs = self.sort_by_text()
        self.logger.info("done with sorting by text, now running transform docs")
        df = pd.DataFrame.from_dict(docs, orient='index').reset_index()

        total_columns = len(df.columns)
        header = list(range(total_columns))
        header[0] = "item_id"
        header[1::2] = ["related_item_" + str(x) for x in range(1, (total_columns + 1) // 2)]
        header[2::2] = ["related_score_" + str(x) for x in range(1, (total_columns + 1) // 2)]
        header = tuple(header)
        df.columns = header
        self.logger.info("this is inside transform_docs, below is shape of item docs")
        self.logger.info(df.shape)
        return df

    def find_synonyms(self):
        self.filter_data()
        self.related_items()
        return self.transform_docs()


