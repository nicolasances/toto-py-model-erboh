FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN pip install --upgrade pip
RUN pip install joblib
RUN pip install pandas
RUN pip install sklearn
RUN pip install gunicorn
RUN pip install toto-pubsub-nicolasances
RUN pip install toto-logger-nicolasances
RUN pip install requests
RUN pip install uuid
RUN pip install google-cloud-pubsub
RUN pip install google-cloud-storage
RUN pip install mlflow
RUN pip install apscheduler

COPY . /app/

WORKDIR /app/

ENV TOTO_TMP_FOLDER=/modeltmp
ENV PYTHONUNBUFFERED=TRUE

CMD gunicorn --bind 0.0.0.0:8080 wsgi:app --enable-stdio-inheritance