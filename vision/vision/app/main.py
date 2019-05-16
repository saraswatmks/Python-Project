"""

This script contains the vision algorithm to recommend similar items.

"""

import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from vision.helpers import get_item_vals, get_param_value, load_stopwords, process_query
from vision.vision import Vision
from vision.s3 import S3
import newrelic.agent as newrelic
from aiohttp import web, ClientSession
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

cache = dict()


class VisionApp(Vision, S3):

    def __init__(self, config_file):
        Vision.__init__(self, config_file)
        S3.__init__(self)
        self.conn = None
        self.session = ClientSession()
        self.mode_dict = {"DEV": None,
                          "STG": {'X-Panamera-Host': "api.olx.co.za",
                                  'Content-Type': "application/json",
                                  'x-origin-panamera': "staging",
                                  'Cache-Control': "no-cache"},
                          "PRD": {'X-Panamera-Host': "api.olx.co.za",
                                  'Content-Type': "application/json",
                                  'x-origin-panamera': "production",
                                  'Cache-Control': "no-cache"}}

    @staticmethod
    async def make_dict(l):

        j = list(l)[1:]
        j = dict(zip(j[::2], j[1::2]))
        j = {k: v for k, v in j.items() if str(k) != "None"}
        return j

    def get_db(self):
        """
        Setup the DATABASE connection in multi-threaded mode.
        """
        logger.info('opening DATABASE connection')

        p = Path(self.config.app.database)
        if not p.exists():
            raise FileNotFoundError("Vision DB Not Found")

        eng = create_engine('sqlite:///{}'.format(self.config.app.database),
                            connect_args={'check_same_thread': False},
                            poolclass=StaticPool)
        conn = eng.connect()
        conn.row_factory = sqlite3.Row

        logger.info('done loading connection')

        self.conn = conn

    async def get_item_json(self, item_id, api_url, mode):

        item_url = api_url + item_id

        async with self.session.get(url=item_url, headers=self.mode_dict[mode]) as resp:
            if resp.status == 200:
                r = await resp.json()
                return r
            else:
                raise web.HTTPBadRequest(text="Item information not available for %s" % item_id)

    async def find_similar_items(self, stop_words, item=None):
        """
        This is the main recommendation algorithm used to find similar items given a new item.
        The cosine distance between item embedding is used as a measure to calculate item similarity.

        conn = active sqlite DATABASE connection object
        item = item id, type: str
        """

        global cache

        # check in the cache
        if item in cache:
            logger.info("found inside cache")
            return cache[item]

        # get following information for the new item
        item_query = "select * from item_docs where item_id = ?"
        f1 = self.conn.execute(item_query, (item,)).fetchone()

        # head items
        if f1 is not None:
            j = await VisionApp.make_dict(f1)
            op = {"match_type": "head_item", "similar_items": j}
            # cache[item] = op
            return op

        # for cold items
        logger.info("this is item %s", item)

        # c_time = time.time()
        # item_json = await self.get_item_json(item_id=item, api_url=self.config.app.api_url, mode=self.config.app.mode)
        item_json = await self.get_item_json(item_id=item, api_url=self.config.app.api_url, mode=self.config.app.mode)
        head_item = await get_item_vals(item_json=item_json)

        # logger.info("time taken to get info from core: %s", time.time() - c_time)
        head_param = await get_param_value(head_item["category"], head_item["params"])

        # do attribute matching
        if head_item["category"] in ["378", "801", "379", "363", "388", "367"]:
            # item attributes matching
            logger.info("doing attribute matching")
            logger.info("this is category_attr %s", (head_item["category"], head_param))

            query = f"""SELECT item_id
                        FROM item_params
                        where cat_attr = ?
                        ORDER BY (({head_item['lat']}-lat)*({head_item['lat']}-lat)) + 
                                 (({head_item['lon']} - lon)*({head_item['lon']} - lon))
                        """
            nearest_item = self.conn.execute(query, (head_param,)).fetchone()

            if nearest_item is not None:

                logger.info(f"this is nearest item: {nearest_item}")

                f3 = self.conn.execute(item_query, (nearest_item,)).fetchone()

                logger.info(f"this is before op: {f3}")

                op = {"match_type": "item_attributes filter",
                      "similar_items": await VisionApp.make_dict(l=f3)}
                cache[item] = op
                return op

        logger.info("breaking out of attribute matching, falling back to category city matching")
        # do city_cat matching
        city_cat = head_item["city"] + "_" + head_item["category"]
        head_title = await process_query(head_item["title"], stop_words)

        logger.info("this is head title %s", head_title)

        db_name = "item_cc_" + city_cat
        bm_query = f""" select item_id
                            from {db_name}
                            where {db_name} MATCH ?
                            order by rank
                            """
        logger.info("check if table exists in the database")

        check = self.conn.execute(f"select name from sqlite_master where name = '{db_name}'").fetchone()

        if check is not None:

            # found items matching the same city and category
            logger.info("doing category city matching with bm25")
            logger.info("this is db_name: %s", db_name)

            f4 = self.conn.execute(bm_query, (head_title,)).fetchone()

            if f4 is not None:
                bm_item = list(f4)[0]
                logger.info("this is the best bm item: %s", bm_item)
                f5 = self.conn.execute(item_query, (bm_item,)).fetchone()
                op = {"match_type": "category city filter",
                      "similar_items": await VisionApp.make_dict(l=f5)}
                cache[item] = op
                return op

        # check matched_items
        logger.info("category city failed, falling back to category matching")

        db_name = "item_cat_" + head_item["category"]
        check = self.conn.execute(f"select name from sqlite_master where name = '{db_name}'").fetchone()
        bm_query = f""" select item_id
                                from {db_name}
                                where {db_name} MATCH ?
                                order by rank
                                """

        if check is not None:
            logger.info("doing only category matching with bm25")

            f6 = self.conn.execute(bm_query, (head_title,)).fetchone()

            if f6 is not None:
                bm_item = list(f6)[0]

                logger.info("this is the best bm item: %s", bm_item)
                logger.info("this is db_name: %s", db_name)
                f7 = self.conn.execute(item_query, (bm_item,)).fetchone()
                op = {"match_type": "category filter",
                      "similar_items": await VisionApp.make_dict(l=f7)}
                cache[item] = op
                return op

        logger.info('doing title cold start across all')
        bm_query = """
                    select item_id
                    from item_titles
                    where item_titles MATCH ?
                    order by rank
                    """
        bm_item = self.conn.execute(bm_query, (head_title,)).fetchone()

        if bm_item is not None:
            bm_item = list(bm_item)[0]
            f8 = self.conn.execute(item_query, (bm_item,)).fetchone()
            op = {"match_type": "all titles",
                  "similar_items": await VisionApp.make_dict(l=f8)}
            cache[item] = op
            return op

        return {"match_type": "nothing matched", "similar_items": None}

    def __call__(self):

        routes = web.RouteTableDef()

        # initialise the connection
        logger.info('loading db in memory')
        self.get_db()

        logger.info('loading stopwords')
        sw = load_stopwords(self.config.app.stopwords)

        @routes.get('/alive')
        async def getalive(request):
            """
            health endpoint
            """
            return web.Response(text="i am alive")

        @routes.get('/ready')
        async def getready(request):
            """
            health endpoint
            """
            return web.Response(text="i am ready")

        @routes.get('/recommend/i2i')
        async def getitem(request):
            """
            API endpoint which returns similar items
            :return: dictionary containing similar items and match type
            """
            item = request.rel_url.query['item']
            items = await self.find_similar_items(stop_words=sw, item=item)

            event_params = {
                'item_id': items,
                'recommended_items': len(items['similar_items']) if items['similar_items'] is not None else 0,
                'match_type': items['match_type']
            }

            newrelic.record_custom_event(event_type="recommend", params=event_params)
            return web.json_response(items)

        app = web.Application()
        app.add_routes(routes)
        return app


async def init_func():
    app = VisionApp(config_file=os.getenv("CONFIG_FILE"))
    return app()


