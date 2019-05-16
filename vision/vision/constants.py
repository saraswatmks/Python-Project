# API_URL = "https://api.olx.co.za/api/v1/items/"
API_URL = "http://panamera-core.panamera-core.svc.cluster.local/api/v1/items/"

DATABASE_PATH = "recommend2.db"

S3_DB_PATH = "s3://stg-relevance-vision-eu-west-1/vision-db/recommend2.db"
# S3_DB_PATH = "s3://olx-relevance-panamera-west/vision-db/recommend2.db"

STOPWORDS_FILE = "resources/STOPWORDS.txt"
ITEM_DOCS_PATH = "s3_path"
ITEM_PARAMS_PATH = "s3_path"
CITY_DATA_PATH = "s3_path"

# Absolute path for the S3 bucket
S3_PARENT_PATH = 's3://olx-relevance-panamera'

# Default model Parameters
# Default number of epochs
DEFAULT_ITERATIONS = 30
# Default version of the model
DEFAULT_VERSION = 1
# Default window size for the model
DEFAULT_WINDOW = 2
# Default size of the embeddings
DEFAULT_SIZE = 50
# Default number of negative samples
DEFAULT_NEG_SAMPLES = 3
# Default minimum count of query
DEFAULT_MIN_COUNT = 10
# Default flag for Skip-gram version of word2vec
DEFAULT_SKIPGRAM_FLAG = 1
# Default random seed
DEFAULT_SEED = 110
# Learning rate to be used for word2vec training
DEFAULT_LEARNING_RATE = 0.01
