

try:
    import pyspark
except ImportError:
    import findspark
    findspark.init()

from pyspark.sql import SparkSession
import os
import sys
import zipfile
from vision.vision import VisionLogger


class Spark(VisionLogger):
    def __init__(self, is_local=False):

        VisionLogger.__init__(self)

        if is_local:
            self.logger.info("starting spark Locally")
            os.environ['PYSPARK_PYTHON'] = "/anaconda3/envs/vision_env/bin/python"

            self.spark = (SparkSession
                          .builder
                          .master("local")
                          .appName('Vision')
                          .getOrCreate())

        else:

            files = [path for path in os.listdir(".") if path.endswith('.pex') and os.path.isfile(path)]
            if len(files) < 1:
                self.logger.error("no pex file found in current directory")
                sys.exit(-1)
            if len(files) > 1:
                self.logger.error("found more than one pex in current directory")
                sys.exit(-1)

            pex_file = os.path.basename(files[0])
            with zipfile.ZipFile(pex_file, "r") as zip_ref:
                zip_ref.extract('STOPWORDS.txt', '.')
                self.logger.info("done extracting zip")

            self.logger.info("starting spark on yarn")
            self.logger.info(("spark home {}".format(os.environ['SPARK_HOME'])))
            os.environ['PYSPARK_PYTHON'] = "./" + pex_file
            os.environ['PYSPARK_DRIVER_PYTHON'] = "./" + pex_file
            os.environ['PEX_ROOT'] = "./.pex"
            self.spark = (SparkSession
                          .builder
                          .master("yarn")
                          .appName('Vision')
                          .config("spark.submit.deployMode", "client")
                          .config("spark.yarn.dist.files", pex_file)
                          .config("spark.executorEnv.PEX_ROOT", "./.pex")
                          .config("spark.executor.instances", 15)
                          .config("spark.driver.memory", "24g")
                          .config("spark.driver.cores", 24)
                          .config("spark.driver.memoryOverhead", "1g")
                          .config("spark.executor.memory", "3g")
                          # .config("yarn.nodemanager.vmem-check-enabled", False)
                          .config("spark.executor.cores", 4)
                          .config("spark.executor.memoryOverhead", "1g")
                          .config("spark.driver.maxResultSize", "10g")
                          .getOrCreate()
                          )
