# CPU-only character extraction service. It never loads a model itself - all
# generation happens in text-generation-webui via the internal API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py core.py backend.py app.py ./
COPY prompts ./prompts
COPY schemas ./schemas

# Matches the securityContext of the Kubernetes deployment (uid/gid 1000).
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin extractor
USER 1000

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
