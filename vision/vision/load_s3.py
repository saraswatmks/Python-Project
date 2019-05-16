import os
from vision.vision import Vision
from vision.s3 import S3


class StartupS3(Vision, S3):
    def __init__(self, config_file):
        Vision.__init__(self, config_file)
        S3.__init__(self)

    def load_s3(self):
        self.download(s3_file_path=self.config.app.s3_db_path, save_path=self.config.app.database)


if __name__ == "__main__":
    obj = StartupS3(config_file=os.getenv("CONFIG_FILE"))
    obj.load_s3()
