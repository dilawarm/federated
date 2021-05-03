FROM python:3.8
ENV PYTHONUNBUFFERED 1
WORKDIR /federated
COPY requirements.txt /federated/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /federated