import unittest
from vision.helpers import load_stopwords, process_query
import asyncio


class TestVision(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        stopwords_file_path = "vision/docs/STOPWORDS.txt"
        cls.stopwords = load_stopwords(stopwords_file_path)

    def test_clean_query(self):

        examples = ["audi 2.0 2009", "new sofa on sale.", " apple iphone ? "]
        actual = ["audi OR 20 OR 2009", "sofa", "apple OR iphone"]
        loop = asyncio.get_event_loop()
        result = [loop.run_until_complete(process_query(x, self.stopwords)) for x in examples]

        self.assertEqual(actual, result)
