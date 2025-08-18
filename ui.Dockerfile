FROM python:3.11-slim
WORKDIR /app
COPY services/ui/requirements.txt services/ui/requirements.txt
RUN pip install --no-cache-dir -r services/ui/requirements.txt
COPY . .
CMD ["streamlit", "run", "services/ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
