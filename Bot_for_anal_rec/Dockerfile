FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md Main.py ./
COPY src ./src
COPY config ./config
COPY data/input/README.md ./data/input/README.md
COPY data/samples ./data/samples

RUN pip install --upgrade pip && pip install -e .

ENTRYPOINT ["python", "Main.py"]
CMD ["--help"]
