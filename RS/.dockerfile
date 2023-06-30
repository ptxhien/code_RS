FROM python:3.9-alpine

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -U flask-cors &&\
  pip install -U scikit-learn $$\
  pip install pandas &&\
  pip install SQLAlchemy &&\
  pip install PyMySQL 

COPY . .

CMD [ "python", "main.py" ]