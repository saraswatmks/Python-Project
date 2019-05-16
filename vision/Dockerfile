FROM continuumio/miniconda3:latest

# ENV MODE=dev

COPY . vision/
WORKDIR vision/
# RUN mkdir /config
# COPY config/app/$MODE/vision_app.ini /config/vision_app.ini
# COPY config/app/$MODE/newrelic.ini /config/newrelic.ini
COPY vision/load_s3.py load_s3.py

COPY scripts/start.sh  start.sh

RUN chmod +x start.sh

# install numpy antes de todos requirements porque este es una dependencia para fastparquet
RUN apt-get update && apt-get install -y \
    build-essential zip && pip install numpy && \
    pip install -r requirements_app.txt

EXPOSE 5000
ENV CONFIG_FILE=/config/vision_app.ini


CMD ["./start.sh"]
