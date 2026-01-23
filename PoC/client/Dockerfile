FROM python:3.10

WORKDIR /app/PoC/client

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
ENV PYTHONUNBUFFERED=1

COPY client/requirements.txt /app/PoC/client/requirements.txt
RUN pip install --no-cache-dir -r /app/PoC/client/requirements.txt

COPY client /app/PoC/client
COPY contract /app/PoC/contract

CMD ["python", "client.py"]
