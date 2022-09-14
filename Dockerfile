FROM python:3.8.12-buster

COPY project_cancer_detection / project_cancer_detection
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn project_cancer_detection.api.fast:app --host 0.0.0.0 --port $PORT
