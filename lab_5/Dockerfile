FROM python:3.12-slim

WORKDIR /app

# Copy files
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY server.sh /app/
RUN chmod +x /app/server.sh

ENV GOOGLE_APPLICATION_CREDENTIALS='/app/secrets/credentials'

EXPOSE 8080

# Ensure server.sh runs with sh
ENTRYPOINT ["sh", "/app/server.sh"]
