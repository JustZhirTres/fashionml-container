# python tensorflow
FROM tensorflow/tensorflow:2.12.0

# req setup
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# code copy
COPY app/ /app
WORKDIR /app

# launch API
CMD ["python", "api.py"]
