"""
GensimS2v class file to facilitate easy training of models.
"""
import gc
import logging
import os

import gensim
import pandas as pd
from datetime import datetime
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.utils import save_as_line_sentence
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Check c compiler,
# More here: https://stackoverflow.com/questions/39781812/how-can-i-tell-if-gensim-word2vec-is-using-the-c-compiler
from vision.constants import DEFAULT_LEARNING_RATE, DEFAULT_SEED

assert gensim.models.doc2vec.FAST_VERSION > -1

logger = logging.getLogger(__name__)

# Constants for the model, to be changed less often
# Frequency of saving the model
SAVE_FREQ = 5000
# Number of workers to be used by gensim for training
WORKERS = os.getenv('WORKERS_COUNT', 5)

if not WORKERS:
    import multiprocessing

    WORKERS = multiprocessing.cpu_count()

logger.info('GOING TO USE A TOTAL OF %s WORKERS', WORKERS)


class GensimS2v:
    """
    GensimS2v class:
        For easy training of models.

    Functionality covered:
        _all_paths_s3 : Collects all objects from s3 buckets and returns paths to documents at DATA_PATH.
        _read_data : Reads data from all_paths variable and shuffles the data by taking sample of all rows in random
                    order.
        SentenceGenerator : SentencesGenerator class for iterating over documents to be used by Gensim.
        EpochSaver : Callback to save model after every epoch.
        EpochLogger : Log information about training, reports time for epochs.
        train : Function for training the word2vec model.
        generate_embeddings: Generate embeddings of the model.
    """

    def __init__(self, documents, log_dir):
        """
        Constructor of the class, initializes the log directory and data directory for training. Reads data
        from S3/data_dir and assigns documents to the class variable.


        :param data_dir: type str directory, data directory
        :param log_dir: type str directory, refers to the log directory to be used for writing logs and saving models
        :returns object of the GensimS2v class.
        """
        self._log_dir = log_dir
        self._documents = documents.sample(frac=1.0)
        self.save_path = "sentences.txt"
        self.save_document()

        if self._documents is None:
            raise ValueError('No documents to process')

        # self._documents = self._documents['docs'].str.split(',').tolist()

    class SentencesGenerator(object):
        """
        SentencesGenerator class for iterating over documents to be used by Gensim.
        """

        def __init__(self, sentences):
            """
            Constructor of the class.
            :param sentences: type list of lists, documents for training
            """
            self._sentences = sentences

        def __iter__(self):
            """
            Function to be used by Gensim for iterating.
            :return: yields sentences one by one
            """
            for sent in self._sentences:
                yield sent

    class EpochSaver(CallbackAny2Vec):
        """Callback to save model after every epoch"""

        def __init__(self, path_prefix, log_dir, save_freq=50):
            """
            Constructor for the class to save models.
            :param path_prefix: type str, specify the prefix to be used while saving model
            :param log_dir: type str directory, specify the log directory (to be created before training)
            :param save_freq: type int, specify the epoch frequency (ex: after 50 epochs) when the model is saved
            """
            self._path_prefix = path_prefix
            self._epoch = 1
            self._log_dir = log_dir
            self._save_freq = save_freq

        def on_epoch_end(self, model):
            """
            The function to save model as 'self._path_prefix'+'_epoch'+'N' where N is number of epoch when it is saved.
            :param model: type object of Gensim word2vec, Model to be saved
            """
            if self._epoch % self._save_freq == 0:
                output_path = '{}_epoch{}.model'.format(self._path_prefix, self._epoch)
                logger.info("Save model to {}".format(os.path.join(self._log_dir, output_path)))
                model.save(os.path.join(self._log_dir, output_path))

            self._epoch += 1

    class EpochLogger(CallbackAny2Vec):
        """
        Log information about training, reports time for epochs.
        """

        def __init__(self):
            """
            Constructor for the class to log progress information.
            """
            self._epoch = 1
            self._start = datetime.now()
            self._end = datetime.now()

        def on_epoch_begin(self, _):
            """
            Print progress information, initializes start time.
            :param _: type gensim word2vec, signature to match the function to be used by gensim
            """
            self._start = datetime.now()
            logger.info("Epoch #%s start", str(self._epoch))

        def on_epoch_end(self, model):
            """
            Print time to for epoch
            :param model: type gensim word2vec, signature to match the function to be used by gensim
            """
            self._end = datetime.now()
            logger.info("Epoch #%s end in %s time", str(self._epoch), str(self._end - self._start))
            logger.info("Epoch #%s, loss %f", str(self._epoch), model.get_latest_training_loss())
            self._epoch += 1
            gc.collect()

    def save_document(self):
        logger.info("converting documents to list")
        docs = [x.split(',') for x in tqdm(self._documents['docs'])]

        # docs = [' '.join(x) for x in tqdm(docs)]
        logger.info("saving sentences to disk")
        save_as_line_sentence(docs, self.save_path)
        # with open(self.save_path, "w") as f:
        #    f.write("\n".join(docs))
        logger.info("finished saving document")

    def train(self, iterations, window, size, negative, min_count, sg):
        """
        Function for training the word2vec model.
        :param iterations: type int, Number of epochs for training
        :param window: type int, window size to define context of word2vec
        :param size: type int, embedding size
        :param negative: type int, number of negative samples
        :param min_count: type int, minimum frequency of word to generate its embedding
        :param sg: type int, skip_gram then 1, else 0
        """
        # sentences = self.SentencesGenerator(self._documents)
        epoch_saver = self.EpochSaver('tmp', self._log_dir, save_freq=SAVE_FREQ)
        epoch_logger = self.EpochLogger()
        self._w2v_model = Word2Vec(corpus_file=self.save_path
                                   , iter=iterations
                                   , window=window
                                   , min_count=min_count
                                   , size=size
                                   , workers=WORKERS
                                   , sg=sg
                                   , seed=DEFAULT_SEED
                                   , negative=negative
                                   , alpha=DEFAULT_LEARNING_RATE
                                   , callbacks=[epoch_saver, epoch_logger]
                                   , compute_loss=True)

    def generate_embeddings(self):
        """
        Generate embeddings of the model with the file (parquet) format as :
        'Word,embedding1,embedding2,embedding3,.....embedding_size'
        Then saves the model to the model_path, supports s3 and local.
        :param model_path: path to save the embeddings
        """
        indexes = self._w2v_model.wv.index2word
        embedding = self._w2v_model.wv.vectors
        size = self._w2v_model.wv.vectors.shape[1]
        # Create a dataframe from embeddings
        e1 = pd.DataFrame(normalize(embedding, axis=1))
        # Assign queries as index column
        e1['item_id'] = indexes

        # Select relevant columns containing the string and the number of columns as the size of embedding
        e1 = e1[['item_id'] + list(range(0, size))]
        # Assign names to the embedding columns
        temp = ['item_id']+['embedding' + str(i) for i in list(range(0, size))]
        e1.columns = tuple(temp)
        logger.info('success')
        return e1

