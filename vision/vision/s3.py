from pathlib import Path
import os
import boto3
import s3fs
from botocore.client import Config
from vision.vision import VisionLogger
from vision.helpers import ProgressPercentage
import fastparquet
import glob
import pandas as pd


class S3(object):
    def __init__(self):
        self._S3_PATH_PREFIX = 's3://'
        self.s3_client = boto3.client('s3', **self.client_params())
        self.logger = VisionLogger().get_logger()

    def client_params(self):
        """
        Fetches the S3 credentials (and endpoint) from environment and returns a map with all present values.
        You should use this method when creating the client, like: `aws.client('s3', **s3_config())`
        :return: a dict with all present S3 credentials/config from environment
        """
        config = self._internal_client_config()
        config['config'] = Config(**self._internal_config_kwargs())

        return config

    def _internal_config_kwargs(self):
        config_kwargs = {}
        version = os.getenv('AWS_S3_SIGNATURE_VERSION')
        if version:
            config_kwargs = {'signature_version': version}
        return config_kwargs

    def _internal_client_config(self):
        config = {
            'endpoint_url': os.getenv("S3_ENDPOINT"),
            'aws_access_key_id': os.getenv("AWS_ACCESS_KEY_ID"),
            'aws_secret_access_key': os.getenv("AWS_SECRET_ACCESS_KEY"),
            'aws_session_token': os.getenv("AWS_SESSION_TOKEN"),
            'region_name': os.getenv("AWS_DEFAULT_REGION")
        }
        # remove keys with None values
        config = {key: value for key, value in config.items() if value}
        return config

    def _s3fs_client_kwargs(self):
        """
        Get client keyword arguments used by s3fs
        :return:
        """
        return self._internal_client_config()

    def _s3fs_config_kwargs(self):
        """
        Get config keyword arguments used by s3fs
        :return:
        """
        return self._internal_config_kwargs()

    def is_s3_path(self, path):
        """
        Returns True if a path refers to a an s3 object/location, and False otherwise
        :param path: type str A path. Ex: s3://bucket/file or /location/file
        :return: type bool. True if path is an s3 path, False otherwise.
        """
        return str(path or "").startswith(self._S3_PATH_PREFIX)

    def extract_bucket_prefix(self, s3_path):
        """
            Extracts the bucket and the prefix/key from an s3 path. Raises ValueError if s3_path is not a valid s3 path.
            Eg: s3://bucket/dir/file will return (bla, dir/file)
            :param s3_path: type str A path. Ex: s3://bucket/file
            :return: type (str, str). A tuple with first value as the s3 bucket and the second one as the key/prefix.
        """
        if not self.is_s3_path(s3_path):
            raise ValueError("Path is not a valid s3 path")

        # using split here instead of lstrip, lstrip fails with stg-paths
        s3_path = str(s3_path).split(self._S3_PATH_PREFIX)[1]
        parts = s3_path.split('/', 1)

        return parts[0], parts[1]

    def get_custom_opener(self):
        """
        Creates a s3fs file opener
        :return:
        """
        custom_open = self.get_s3fs().open
        return custom_open

    def get_s3fs(self):
        """Get S3FS correctly configured"""
        return s3fs.S3FileSystem(client_kwargs=self._s3fs_client_kwargs(),
                                 config_kwargs=self._s3fs_config_kwargs())

    def get_matching_s3_keys(self, bucket, prefix='', suffix=''):
        """
        Generate the keys in an S3 bucket.

        :param bucket: type str, name of the S3 bucket.
        :param prefix: type str, only fetch keys that start with this prefix (optional).
        :param suffix: type str, only fetch keys that end with this suffix (optional).
        """
        kwargs = {'Bucket': bucket}
        keys = []
        # If the prefix is a single string (not a tuple of strings), we can
        # do the filtering directly in the S3 API.
        if isinstance(prefix, str):
            kwargs['Prefix'] = prefix
        while True:
            # The S3 API response is a large blob of metadata.
            # 'Contents' contains information about the listed objects.
            resp = self.s3_client.list_objects_v2(**kwargs)
            if 'Contents' in resp:
                for obj in resp['Contents']:
                    key = obj['Key']
                    if key.startswith(prefix) and key.endswith(suffix):
                        keys.append(key)
            # The S3 API is paginated, returning up to 1000 keys at a time.
            # Pass the continuation token into the next response, until we
            # reach the final page (when this field is missing).
            try:
                kwargs['ContinuationToken'] = resp['NextContinuationToken']
            except KeyError:
                break
        absolute_paths = [(bucket, path) for path in keys]

        return absolute_paths

    def download(self, s3_file_path, save_path):

        p = Path(save_path)

        if p.exists():
            self.logger.info("file found in the directory")
            return
        self.logger.info('file not found, downloading...')

        bucket, key = self.extract_bucket_prefix(s3_file_path)

        self.logger.info("this is bucket name: {}".format(bucket))
        self.logger.info("this is key: {}".format(key))

        self.s3_client.download_file(Bucket=bucket,
                                     Key=key,
                                     Filename=save_path,
                                     Callback=ProgressPercentage(self.s3_client, bucket, key))

    def upload(self, local_file, s3_path):
        if os.path.isfile(local_file):
            bucket, key = self.extract_bucket_prefix(s3_path)

            self.logger.info("this is bucket name: {}".format(bucket))
            self.logger.info("this is key: {}".format(key))

            self.logger.info("file found, uploading file {} to {}...".format(local_file, '/'.join([bucket, key])))
            self.s3_client.upload_file(local_file, bucket, key)

    def write_parquet(self, dataframe, path, copression='gzip'):
        fast_parquet_kwargs = {"compression": copression}
        if self.is_s3_path(path):
            fast_parquet_kwargs['open_with'] = self.get_custom_opener()
        fastparquet.write(path, dataframe, **fast_parquet_kwargs)

    def read_parquet(self, path):
        if self.is_s3_path(path):
            bucket, prefix = self.extract_bucket_prefix(path)
            all_paths = self.get_matching_s3_keys(bucket, prefix, 'parquet')
            self.logger.info('Found a total of %d files to read in s3', len(all_paths))
            all_paths = ['s3://' + bucket + '/' + path for (bucket, path) in all_paths]
            return fastparquet.ParquetFile(all_paths, open_with=self.get_custom_opener()).to_pandas()
        else:
            files = [os.path.join(path, f) for f in glob.glob(os.path.join(path, "*.parquet"))]
            if len(files) > 0:
                return fastparquet.ParquetFile(files).to_pandas()
            return pd.DataFrame()
