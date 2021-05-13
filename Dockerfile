FROM python:3.9.1

WORKDIR /src/

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY filtered_stochastic_newton.py .

# command to run on container start
ENTRYPOINT [ "python", "./filtered_stochastic_newton.py" ]
