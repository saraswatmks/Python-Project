#!/usr/bin/env bash

echo "starting vision app...."
echo "downloading db"
python ./load_s3.py

echo "starting service"
NEW_RELIC_CONFIG_FILE=/config/newrelic.ini newrelic-admin run-program gunicorn -k aiohttp.worker.GunicornWebWorker -c vision/gunicorn_conf.py vision.app.main:init_func