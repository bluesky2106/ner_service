FROM --platform=linux/amd64 python:latest as build

RUN mkdir -p /usr/src/app/backend

WORKDIR /usr/src/app/backend

COPY . /usr/src/app/backend

# Install Java environment to run VnCoreNLP
RUN apt-get update && \
    apt-get install -y openjdk-17-jre-headless && \
    apt-get install -y openjdk-17-jdk-headless && \
    apt-get clean;

RUN pip install -U -r requirements.txt

CMD python app.py