FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY raggiroti /app/raggiroti
COPY rulebook /app/rulebook
COPY data /app/data

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "raggiroti.web.app:app", "--host", "0.0.0.0", "--port", "8080"]

