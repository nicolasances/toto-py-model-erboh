FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN pip install --upgrade pip
RUN pip install joblib
RUN pip install pandas
RUN pip install sklearn
RUN pip install gunicorn
RUN pip install toto_pubsub
RUN pip install toto_logger
RUN pip install requests
RUN pip install uuid


COPY . /app/

WORKDIR /app/

ENV TOTO_TMP_FOLDER=/modeltmp

CMD gunicorn --bind 0.0.0.0:8080 wsgi:app