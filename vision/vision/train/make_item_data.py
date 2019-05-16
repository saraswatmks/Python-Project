"""
This script returns a spark data frame contains item level information processed from reservoir ads data.
"""

try:
    import pyspark
except ImportError:
    import findspark
    findspark.init()

import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql.window import *
from vision.train.spark import Spark

PARAM_COL_NAME = "params"
PARAMS_MAP_NAME = "params_map"


class ItemData(Spark):
    def __init__(self,
                 ads_paths,
                 local=False):
        Spark.__init__(self, is_local=local)
        self.ads_paths = ads_paths
        self.ads = None

    def load_ads_data(self):

        self.logger.info('this is the input ad path')
        self.logger.info(self.ads_paths)
        self.logger.info("this is path type: %s" % type(self.ads_paths))

        self.ads = self.spark.read.json(self.ads_paths)
        self.logger.info('ads data loading done')

    def remove_duplicate_ads(self):

        self.ads = (self.ads
                    .withColumn("row_num", f.row_number().over(Window.partitionBy("id").orderBy(f.desc("created_at"))))
                    .filter("row_num = 1")
                    .select('id',
                            'created_at',
                            'city_id',
                            'status',
                            'category_id',
                            'params',
                            'title',
                            'description',
                            'map_lat',
                            'map_lon'))

        self.logger.info("done removing duplicate ads and select columns")

    @staticmethod
    def strange_to_json(paramstring):

        if paramstring is not None:
            try:
                if "<=>" not in paramstring:
                    return paramstring
                else:
                    t = paramstring.replace(":", "")
                    t = t.split("<br>")
                    t = [x for x in t if "<=>" in x]
                    t = [x.replace('<=>', ':') for x in t]
                    return dict([x.split(':') for x in t])
            except Exception as e:
                print('this is the error')
                print(paramstring)
                raise e
        else:
            return paramstring

    @staticmethod
    def schema_params():

        item_attributes_schema = StructType([
            StructField("model", StringType(), True),
            StructField("mileage", StringType(), True),
            StructField("make", StringType(), True),
            StructField("bathrooms", StringType(), True),
            StructField("bedrooms", StringType(), True),
            StructField("property_type", StringType(), True),
            StructField("brand", StringType(), True),
            StructField("brand_model", StringType(), True)
        ])

        json_params = f.from_json(
            f.regexp_replace(
                f.regexp_replace(
                    f.regexp_replace("params", "\"\"", "\""), "\"\\{", "\\{"),
                "\\}\"", "\\}"),
            item_attributes_schema)

        return item_attributes_schema, json_params

    @staticmethod
    def select_cols(data):
        data = data.select("id",
                           "created_at",
                           "status",
                           "category_id",
                           "city_id",
                           "title",
                           "description",
                           "map_lat",
                           "map_lon",
                           "model",
                           "mileage",
                           "make",
                           "bathrooms",
                           "bedrooms",
                           "property_type",
                           "brand",
                           "brand_model")
        return data

    @staticmethod
    def get_attributes(data):

        data = (data.withColumn("model", f.col("params_map.model"))
                .withColumn("mileage", f.col("params_map.mileage"))
                .withColumn("make", f.col("params_map.make"))
                .withColumn("bathrooms", f.col("params_map.bathrooms"))
                .withColumn("bedrooms", f.col("params_map.bedrooms"))
                .withColumn("property_type", f.col("params_map.property_type"))
                .withColumn("brand", f.col("params_map.brand"))
                .withColumn("brand_model", f.col("params_map.brand_model")))

        return data

    @staticmethod
    def final_clean_up(data):

        """
        Discovered that even after remove_duplicates there are bogus (title, single numbers) item_id
        values left in item_attributes table. Removing them here.
        :param data:
        :return: data
        """

        data = data.filter(f.col("id").cast("int").isNotNull())
        data = data.filter("map_lat is not null")
        data = data.filter(f.length(f.col('id')) > 8)
        return data

    def process_ads(self):
        self.load_ads_data()
        self.remove_duplicate_ads()
        item_schema, json_params = ItemData.schema_params()
        strange_json_udf = f.udf(ItemData.strange_to_json)
        strange_params = f.from_json(strange_json_udf(PARAM_COL_NAME), item_schema)
        self.ads = self.ads.withColumn(PARAMS_MAP_NAME, f.coalesce(json_params, strange_params))
        self.ads = ItemData.get_attributes(self.ads)
        self.ads = ItemData.select_cols(self.ads)
        self.ads = ItemData.final_clean_up(self.ads)
        self.logger.info("done creating item data, moving to saving")
        return self.ads
