import csv
import re
import threading
import sys


class ProgressPercentage(object):

    def __init__(self, resource, bucket, key):
        self._filename = key
        self._size = resource.head_object(Bucket=bucket, Key=key)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


def load_stopwords(stopwords_file):
    """
    Function to load stopwords from nltk and custom stopwords file.
    :return:
    """

    custom_sw = csv.reader(open(stopwords_file, 'r'))
    custom_sw = [row for row in custom_sw]
    custom_sw = [i for s in custom_sw for i in s]

    return custom_sw


async def process_query(query, stopwords):
    """
    Clean the input text by removing everything but letters, numbers, and spaces.
    Also it remove dots(.) which do not occur between two digits.
    Ex. 2.3, 4.5 are kept as is.

    :param query: search query or item title, type: str
    :param stopwords: set of stopwords, type: set or list
    :return: a sqlite friendly search query

    Examples:
    query = "honda city 2.5 very new."
    output= "honda OR city OR 2.5 OR honda city 2.5"
    """

    # replaces everything except letters, numbers, dots and whitespaces with a whitespace
    query = re.sub(r'[^a-zA-Z0-9.\s ]+', ' ', query)
    query = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', query)
    query = re.sub(r'\.', r'', query)
    query = re.sub(r'\s\s+', ' ', query.lower().strip())

    nq = ' OR '.join([x for x in query.split(' ') if x not in stopwords])
    return nq


async def get_item_vals(item_json):
    """
    Retrieves item information from the api

    :param item_json: item id, type: str
    """

    title = item_json['data']['title']
    category = item_json['data']['category_id']
    location_user = item_json['data']['locations'][0]
    city = location_user['city_id']
    params = item_json['data']['parameters']

    return {'title': title,
            'lat': location_user['lat'],
            'lon': location_user['lon'],
            'params': params,
            'category': category,
            'city': city}


async def get_param_value(category, params):
    """
    Based on pre-defined criteria, extracts item attribute information given an item category.

    :param category: item category, type: str
    :param params: item parameters extract from api, type: a list consisting of dictionaries
    :return: category,attribute information, type:str
    """

    if not params:
        return None

    if category == "378":
        # for cars use model
        return ",".join([category, [x["value"] for x in params if "model" in x.values()][0]])
    elif category in ["363", "388", "367"]:
        # for house on rent
        return ",".join([category, [x["value"] for x in params if "bedrooms" in x.values()][0]])
    elif category == "379":
        # for bikes
        return ",".join([category, [x["value"] for x in params if "make" in x.values()][0]])
    elif category == "801":
        # for mobiles
        return ",".join([category, [x["value"] for x in params if "brand" in x.values()][0]])
    else:
        return ",".join([category, "None"])
