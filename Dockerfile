FROM python:3.12.6-slim

WORKDIR /app
COPY . .

COPY  requirements.txt .

COPY . . 
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn","fast:app","--host","0.0.0.0","--port","8000","--reload"]