FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install numpy==1.26.4 && \
    pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["bash", "start.sh"]