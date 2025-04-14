FROM python:3

WORKDIR /src

RUN pip install flask
RUN pip install flask-mysql

COPY . /src

ENTRYPOINT FLASK_APP=src/app.py flask run

docker build -t simple-app