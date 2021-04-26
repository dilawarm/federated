FROM python:3.9
ENV PYTHONUNBUFFERED 1
WORKDIR /federated
COPY . /federated