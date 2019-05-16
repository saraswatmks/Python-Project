from munch import munchify
import logging
from configobj import ConfigObj


class VisionLogger(object):
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_logger(self):
        return self.logger


class Vision(VisionLogger):
    def __init__(self, config_file):
        VisionLogger.__init__(self)
        self.config = munchify(dict(ConfigObj(config_file, list_values=True)))
