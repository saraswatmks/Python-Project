"""
This script returns a spark dataframe containing item sessions.
"""

from vision.train.spark import Spark


class ItemSessions(Spark):
    def __init__(self,
                 android_paths,
                 web_paths,
                 ads_paths,
                 filter_date,
                 sample=False,
                 local=False):
        self.android_paths = android_paths
        self.web_paths = web_paths
        self.ads_paths = ads_paths
        self.sample = sample
        self.local = local
        self.filter_date = filter_date
        Spark.__init__(self, is_local=local)

    def load_data(self, paths, db_name):

        self.logger.info('this is the input path')
        self.logger.info(paths)
        self.logger.info("this is path type: %s" % type(paths))

        data = self.spark.read.parquet(paths)

        self.logger.info('android data loading done')
        if self.sample:
            data = data.where("session_long like '%88'")
        data.createOrReplaceTempView(db_name)
        self.logger.info(f"loading loading {db_name} data into db")

    def load_ads_data(self, paths, db_name):

        self.logger.info('this is the input ad path')
        self.logger.info(paths)
        self.logger.info("this is path type: %s" % type(paths))

        ads = self.spark.read.json(paths)
        ads.createOrReplaceTempView(db_name)
        self.logger.info('ads data loading done')

    def create_item_sessions(self):
        self.logger.info(f"this is the date: {self.filter_date}")
        query = f"""
        with df3 as (
        with df2 as (
        with all_df as (
        with df1 as (
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
                  from_unixtime(cast(meta_date as integer)) >= cast("{self.filter_date}" as date)
        ), df2 as (
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
                   from_unixtime(cast(meta_date as integer)) >= cast("{self.filter_date}" as date)
        ) SELECT *
            FROM df1
            UNION ALL
          SELECT *
            FROM df2
        ), ads as (
            SELECT 
                id,
                status,
                row_number() over (partition by id order by created_at desc) as rn
          FROM ads_data
          WHERE status in ('new','active','sold') and 
                created_at >= "{self.filter_date}"
        ) SELECT *
            FROM all_df a
            INNER JOIN ads b
            ON a.item_id = b.id
            WHERE b.rn = 1
        ) SELECT *
            FROM df2
            ORDER BY session_long, session, session_seq
        ) SELECT session_long, session, concat_ws(',', collect_list(item_id)) AS docs
            FROM df3
            GROUP BY session_long, session
            HAVING size(collect_set(item_id)) > 1
        """
        return self.spark.sql(query)

    def process_sessions(self):
        self.load_data(paths=self.android_paths, db_name="hydra_android")
        self.load_data(paths=self.web_paths, db_name="hydra_web")
        self.load_ads_data(paths=self.ads_paths, db_name="ads_data")
        item_sessions = self.create_item_sessions()
        self.logger.info("done executing sql query")
        return item_sessions
