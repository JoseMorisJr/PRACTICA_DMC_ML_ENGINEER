FROM apache/airflow:2.9.2
COPY requirements.txt /requirements.txt
COPY dags/data ./opt/airflow/dags/data
RUN python3 -m pip install --upgrade pip
RUN pip install -r /requirements.txt


