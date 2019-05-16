import pandas as pd
from tqdm import tqdm

from vision.s3 import S3
from vision.vision import VisionLogger
from datetime import datetime

import sqlite3
print('this is sqlite3 path:', sqlite3.__path__)


class VisionDB(object):

    def __init__(self, database_s3_path, item_synonyms, item_params, location_data):

        """
        initialises the class with following paths

        :param database_s3_path: file path where the database will be saved, eg: s3://olx-relevance-panamera/item/recommend.db
        :param item_synonyms: a data frame containing item to item synonyms
        :param item_params: a data frame containing item params information, this is different from item attributes
        :param location_data: a data frame contains city and its related items based on similar locations
        """
        self.logger = VisionLogger().get_logger()
        now = datetime.now().strftime("%Y-%m-%d-%H")
        self.local_db_path = f"vision-{now}.db"
        self.conn = sqlite3.connect(self.local_db_path)
        self.nrows = None
        self.item_synonyms = item_synonyms
        self.item_params = item_params
        self.location_data = location_data
        self.database_s3_path = database_s3_path
        self.s3 = S3()

    # Load item data
    def load_item_docs(self):
        self.logger.info('loading item docs')

        # do not remove these comments
        # df = pd.read_csv(self.item_synonyms)
        # df.iloc[:, 1::2] = df.iloc[:, 1::2].applymap(lambda x: str(int(x)) if str(x) != 'nan' else x)
        self.logger.info("saving item docs as dict")
        self.item_synonyms['item_id'] = self.item_synonyms['item_id'].astype(str)
        self.item_synonyms.to_sql("item_docs", self.conn, index=False)
        self.conn.execute("CREATE UNIQUE INDEX idx_item_docs_item_id ON item_docs (item_id);")

        # df = self.item_synonyms.set_index('item_id').T.to_dict(orient="list")
        # df = {k: [i for i in v if str(i) != "nan" or i is not None] for k, v in df.items()}
        # df = {k: str(dict(zip(v[::2], v[1::2]))) for k, v in df.items()}
        # df = pd.DataFrame.from_dict(df, orient='index').reset_index()
        # df.columns = ["item_id", "related"]
        # df.to_sql("item_docs", self.conn, index=False)
        # self.conn.execute("CREATE UNIQUE INDEX idx_item_docs_item_id ON item_docs (item_id);")

        self.logger.info('done loading item docs')
        self.conn.commit()

    def load_item_titles(self):
        self.conn.execute('CREATE VIRTUAL TABLE item_titles USING fts5 (item_id, title);')

        for _, row in self.item_params.iterrows():
            self.conn.execute(f"insert into item_titles (item_id, title) values (?,?)", (row["item_id"], row["title"],))
        self.logger.info('done loading item titles')
        self.conn.commit()

    def load_item_params(self):

        self.item_params['item_id'] = self.item_params['item_id'].astype(str)

        # do not need virtual table for item params
        self.item_params[['item_id', 'lat', 'lon', 'cat_attr']].to_sql('item_params', self.conn, index=False)
        self.conn.execute("CREATE INDEX idx_item_params_cat_attr ON item_params (cat_attr);")
        self.logger.info('done loading item params')
        self.conn.commit()

    def load_item_categories(self):

        all_cats = self.item_params['category_id'].unique()
        data = self.item_params.copy()

        for cat in tqdm(all_cats):
            table = "item_cat_" + str(cat)
            df = data.loc[data["category_id"] == cat, ["item_id", "title"]].reset_index(drop=True)
            self.conn.execute(f"CREATE VIRTUAL TABLE {table} USING fts5 (item_id, title);")
            for _, row in df.iterrows():
                self.conn.execute(f"insert into {table} (item_id, title) values (?,?)", (row["item_id"], row["title"],))
        self.logger.info('done loading item categories data')
        self.conn.commit()

    def load_city_items(self):
        # load item city tables

        all_cc = self.location_data['city_cat'].unique()
        city_data = self.location_data.copy()

        for i, c in tqdm(enumerate(all_cc)):
            table_name = 'item_cc_' + c
            f = city_data.loc[city_data['city_cat'] == c, ['items', 'titles']].reset_index(drop=True)
            f = (pd.concat([f[x].apply(pd.Series).stack() for x in f.columns], axis=1)
                 .reset_index(drop=True)
                 .rename(columns={0: 'item_id', 1: 'title'}))
            # self.logger.info(f"{i}, there are {f.shape} rows in {table_name} city category combination.")
            self.conn.execute(f'CREATE VIRTUAL TABLE {table_name} USING fts5 (item_id, title);')
            for _, row in f.iterrows():
                self.conn.execute(f'insert into {table_name} (item_id, title) values (?,?)', (row['item_id'], row['title'],))
        self.logger.info('done loading city items data')
        self.conn.commit()

    def create(self):
        self.load_item_docs()
        self.load_item_titles()
        self.load_item_params()
        self.load_item_categories()
        self.load_city_items()
        self.conn.close()

    def upload_to_s3(self):
        self.s3.upload(self.local_db_path, self.database_s3_path + self.local_db_path)
        self.s3.upload(self.local_db_path, self.database_s3_path + 'current/vision.db')

