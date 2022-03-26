# use this file with:
# docker build --no-cache -t hibiscus .
# docker run --rm -ti -v $(pwd):/home/felixity hibiscus

FROM python:3.10.2 AS python-dependencies

# install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# switch to non-root user
RUN useradd felixity
USER felixity
WORKDIR /home/felixity

FROM python-dependencies

# copy file to run
COPY filtered_stochastic_newton.py .

# runs when container starts
ENTRYPOINT [ "python", "./filtered_stochastic_newton.py" ]
