FROM python:3.11
WORKDIR /app
RUN apt update

COPY . .

RUN apt-get install git
RUN git config --global init.defaultBranch main

RUN pip install --no-cache -r requirements.txt
