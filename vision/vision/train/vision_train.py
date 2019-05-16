"""
train_model function used by click to train the model.
"""

import click

from vision.db import VisionDB
from vision.s3 import S3
from vision.train.city_titles import CityTitles
from vision.train.gensim_model import GensimS2v
from vision.train.item_params import ItemParams
from vision.train.item_synonyms import ItemSynonyms
from vision.train.make_item_data import ItemData
from vision.train.make_item_sessions import ItemSessions
from vision.train.spark import Spark
from vision.vision import Vision
import pandas as pd


class VisionTrain(Vision, Spark):
    def __init__(self, config_file, is_local):
        self.is_local = is_local

        Vision.__init__(self, config_file)
        Spark.__init__(self, is_local=self.is_local)
        self.item_sessions = None
        self.embeddings = None
        self.synonyms = None
        self.item_attributes = None
        self.item_params = None
        self.city_data = None  # self.s3.read_parquet(self.config.train.location_data_path)
        self.s3 = S3()

    def preprocessing(self):

        processor = ItemSessions(android_paths=self.config.train.android_tracking_paths,
                                 web_paths=self.config.train.web_tracking_paths,
                                 ads_paths=self.config.train.ads_paths,
                                 filter_date=self.config.train.filter_date,
                                 sample=False,
                                 local=self.is_local)

        self.item_sessions = processor.process_sessions()
        # self.item_sessions.cache()
        # self.item_sessions.write.mode("overwrite").parquet(self.config.train.sessions_dir)
        self.logger.info('done generating item sessions')

        item_instance = ItemData(ads_paths=self.config.train.ads_paths,
                                 local=self.is_local)

        self.item_attributes = item_instance.process_ads()
        self.logger.info('saving ads to parquet')
        # self.item_attributes.write.mode("overwrite").parquet(self.config.train.item_data_path)
        self.logger.info('converting parquet ads to pandas')
        self.item_attributes = self.item_attributes.toPandas()
        self.logger.info('done generating ads info')

    def train(self):
        self.logger.info('reading data')
        # self.item_sessions = self.spark.read.parquet(self.config.train.sessions_dir)
        # Initializing the object and reading data
        self.logger.info("converting item sessions to pandas")
        self.item_sessions = self.item_sessions.toPandas()

        self.logger.info('Training')
        s2v = GensimS2v(documents=self.item_sessions, log_dir=self.config.train.log_dir)
        # Train model with the parameters passed to the self
        s2v.train(iterations=int(self.config.train.iters),
                  window=int(self.config.train.window),
                  size=int(self.config.train.size),
                  negative=int(self.config.train.negative),
                  min_count=int(self.config.train.min_count),
                  sg=int(self.config.train.sg))
        self.logger.info('Training done!')
        # Generate embeddings from the trained model
        self.embeddings = s2v.generate_embeddings()
        # self.item_sessions.unpersist()
        if not self.embeddings.empty:
            # self.spark.createDataFrame(self.embeddings).write.mode("overwrite").parquet(self.config.train.embeddings_path)
            self.logger.info('Embeddings generated!')
        else:
            self.logger.warn("Embeddings are empty")

    def generate_synonyms(self):
        self.logger.info('starting with generating item synonyms')
        self.synonyms = ItemSynonyms(
                        item_embeddings=self.embeddings,
                        item_attributes=self.item_attributes,
                        similarity_threshold=float(self.config.train.threshold),
                        date_filter=pd.Timestamp.today() - pd.offsets.MonthBegin(int(self.config.train.embedding_month)),
                        top_related=int(self.config.train.top_similar),
                        distance_threshold_km=int(self.config.train.distance_threshold_km),
                        is_local=self.is_local
                        ).find_synonyms()
        if not self.synonyms.empty:
            self.logger.info("saving synonyms file")
            # self.synonyms.to_csv("synonyms.csv", index=False)
            # self.logger.info("saving item_attributes file")
            # self.item_attributes.to_csv("item_attributes.csv", index=False)
            # self.s3.write_parquet(self.synonyms, self.config.train.synonyms_path)
            # self.spark.createDataFrame(self.synonyms).write.mode("overwrite").parquet(self.config.train.synonyms_path)
        else:
            self.logger.warning("Synonyms are empty")

    def generate_item_params(self):
        self.logger.info('starting with generating final item params')
        self.item_params = ItemParams(item_data=self.item_attributes,
                                      item_docs=self.synonyms,
                                      is_local=self.is_local
                                      ).transform_docs()
        if not self.item_params.empty:
            self.logger.info("saving item params file")
            # self.s3.write_parquet(self.item_params, self.config.train.item_params_path)
            # self.spark.createDataFrame(self.item_params).write.mode("overwrite").parquet(self.config.train.item_params_path)

    def generate_citydata(self):
        self.logger.info('starting with generating city titles data')
        ct = CityTitles(similar_city_docs_path=self.config.train.similar_city_docs,
                        item_docs_df=self.synonyms,
                        item_params_df=self.item_params,
                        item_data_df=self.item_attributes)
        self.city_data = ct.generate_docs()
        # self.logger.info("saving citydata docs")
        # self.city_data.to_csv("city_data.csv", index=False)
        # self.item_params.to_csv("item_params.csv", index=False)
        self.logger.info('done generating city titles data')

    def generate_db(self):
        db = VisionDB(database_s3_path=self.config.train.database_path,
                      item_synonyms=self.synonyms,
                      item_params=self.item_params,
                      location_data=self.city_data
                      )
        db.create()
        # self.item_params.unpersist()
        db.upload_to_s3()

    def test_local(self):
        self.logger.info("starting with local")
        self.embeddings = self.s3.read_parquet(self.config.train.embeddings_path)
        self.item_attributes = self.s3.read_parquet(self.config.train.item_data_path)

    def run(self):
        self.logger.info("starting vision data pipeline")
        self.preprocessing()
        self.train()
        # self.test_local()
        self.generate_synonyms()
        self.generate_item_params()
        self.generate_citydata()
        self.generate_db()


@click.command(short_help='Trains a model')
@click.option("--config-file", required=True,  type=click.Path())
@click.option('--local', is_flag=True)
def train(config_file, local):

    VisionTrain(config_file, is_local=local).run()


if __name__ == '__main__':
    train()
