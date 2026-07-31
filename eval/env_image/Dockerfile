# MerchantBench env server image — used by eval/run_eval.py as the simulation
# backend.
#
# Build from the repo root (build context = env/):
#   docker build -f eval/env_image/Dockerfile -t merchantbench-env:dev env/
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build context is env/, so paths below are relative to it.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY core/      /app/core/
COPY tools/     /app/tools/
COPY storage/   /app/storage/
COPY data/      /app/data/
COPY web/       /app/web/
COPY scenarios/ /app/scenarios/
COPY run.py     /app/run.py

EXPOSE 5000

CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "5000"]
