import re
from pyspark.sql.types import *
from pyspark.sql.window import *
import pyspark.sql.functions as f
from vision.train.spark import Spark
from vision.vision import VisionLogger


class Preprocessing(Spark):
    def __init__(self, android_tracking_paths,
                 web_tracking_paths,
                 ads_paths,
                 sample=False,
                 local=False):
        Spark.__init__(self, local)
        self.logger = VisionLogger().get_logger()
        self.ads_paths = ads_paths
        self.android_tracking_paths = android_tracking_paths
        self.web_tracking_paths = web_tracking_paths
        self.sample = sample
        self.ads = None

    def read_ads(self):
        self.ads = self.spark.read.json(self.ads_paths)
        self.logger.info('data loading done')
        if self.sample:
            self.ads = self.ads.where("session_long like '%88'")
        self.logger.info("done loading data...")

    def read_trackings(self, paths, db_name):
        tackings = self.spark.read.parquet(paths)
        self.logger.info('android data loading done')
        if self.sample:
            tackings = tackings.where("session_long like '%88'")
        tackings.createOrReplaceTempView(db_name)
        self.logger.info(f"loading loading ${db_name} data into db")

    def remove_duplicate_ads(self):
        self.ads = (self.ads
                    .withColumn("row_num", f.row_number().over(Window.partitionBy("id").orderBy(f.desc("created_at"))))
                    .filter("row_num = 1")
                    .select('id',
                            f.to_timestamp(f.col('created_at')).alias("created_at"),
                            'city_id',
                            'status',
                            'category_id',
                            'params',
                            'title',
                            'description',
                            'map_lat',
                            'map_lon'))

        self.logger.info("done removing duplicate ads and select columns")

    def strange_to_json(self, paramstring):
        if "<=>" not in paramstring:
            return paramstring
        else:
            t = re.sub(r'<=>',':',paramstring)
            t = re.sub(r'<br>',',',t)
            return dict([j.split(':') for j in t.split(',')])

    def schema_params(self):

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

    def extract_attributes(self):

        item_schema, json_params = self.schema_params()

        def strange_to_json(paramstring):
            if paramstring is None:
                return ""
            if "<=>" not in paramstring:
                return paramstring
            else:
                t = re.sub(r'<=>', ':', paramstring)
                t = re.sub(r'<br>', ',', t)
                return dict([j.split(':') for j in t.split(',')])

        strange_json_udf = f.udf(strange_to_json)

        self.ads = (self.ads.withColumn("params_map",
                                        f.coalesce(json_params,
                                                   f.from_json(strange_json_udf("params"), item_schema)
                                                   )
                                        )
                    .withColumn("model", f.col("params_map.model"))
                    .withColumn("mileage", f.col("params_map.mileage"))
                    .withColumn("make", f.col("params_map.make"))
                    .withColumn("bathrooms", f.col("params_map.bathrooms"))
                    .withColumn("bedrooms", f.col("params_map.bedrooms"))
                    .withColumn("property_type", f.col("params_map.property_type"))
                    .withColumn("brand", f.col("params_map.brand"))
                    .withColumn("brand_model", f.col("params_map.brand_model"))
                    .select("id",
                            "created_at",
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
                            "brand_model"
                            )
                    )

    def create_item_sessions(self):
        query = f"""
        with df1 as (
        with df as ((
            SELECT
            'android' platform,
            from_unixtime(cast(meta_date as integer)) as time_id,
            meta_session_long as session_long,
            meta_session as session,
            cast(meta_session_count as integer) as session_seq,
            params_item_id  item_id
            FROM hydra_android
            WHERE params_en = 'view_item' AND 
                params_item_id is not null AND 
                lower(params_cc) = 'za' and
                lower(params_extra_item_status) in ('new','active','featured','sold','modify','pending')
        ) UNION ALL(
            SELECT
                'web' platform,
                from_unixtime(cast(meta_date as integer)) as time_id,
                meta_session_long as session_long,
                meta_session as session,
                cast(meta_session_count as integer) as session_seq,
                params_item_id  item_id
            FROM hydra_web
            WHERE params_en = 'view_item' AND 
                  params_item_id is not null AND 
                  lower(params_cc) = 'za' and
                  lower(params_extra_item_status) in ('new','active','featured','sold','modify','pending')
        )) SELECT *
            FROM df
            ORDER BY session_long, session, session_seq
        ) SELECT session_long, session, concat_ws(',', collect_list(item_id)) AS docs
            FROM df1
            GROUP BY session_long, session
            HAVING size(collect_set(item_id)) > 1
        """
        self.item_sessions = self.spark.sql(query)

    def process_item_attributes(self):
        self.read_ads()
        self.remove_duplicate_ads()
        self.extract_attributes()
        return self.ads

    def process_sessions(self):
        self.read_trackings(paths=self.android_tracking_paths, db_name="hydra_android")
        self.read_trackings(paths=self.web_tracking_paths, db_name="hydra_web")
        self.create_item_sessions()
        return self.item_sessions






