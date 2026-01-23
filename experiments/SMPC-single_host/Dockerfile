FROM python:3.10-slim

WORKDIR /app

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

COPY experiments/requirements.txt /app/experiments/requirements.txt
RUN pip install --no-cache-dir -r /app/experiments/requirements.txt

COPY experiments /app/experiments
