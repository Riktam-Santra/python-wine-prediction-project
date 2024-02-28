FROM python:3.9.13

WORKDIR /project

EXPOSE 8080
EXPOSE 5000

COPY . /project/

RUN pip install -r ./requirements.txt
