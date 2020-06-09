FROM python:3.6-slim 

WORKDIR /app

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY ./requirements.txt  .

RUN pip install  --no-cache-dir  -r requirements.txt
RUN python -m nltk.downloader stopwords punkt averaged_perceptron_tagger wordnet

COPY  ./ ./ 

CMD ["gunicorn", "--config", "gunicorn_config.py", "wsgi:app"]