# MerchantBench submission template.
#
# Build from the repo root (build context = agent/):
#   docker build -f agent/submission_template/Dockerfile \
#                -t my-submission:v1 agent/
#
# What this template does:
#   1. Installs agent/requirements.txt (requests + openai + python-dotenv).
#      Add your own deps here if you need them.
#   2. Copies agent/sdk/ — the public SDK every submission talks to.
#   3. Copies agent/submission_template/my_agent.py — the file YOU edit.
#      Replace the stub with your agent. The contract is documented in
#      agent/README.md and summarized in my_agent.py.
#   4. Entrypoint runs your agent. The harness injects:
#         MERCHANTBENCH_BASE_URL  MERCHANTBENCH_RUN_ID  MERCHANTBENCH_AGENT_ID
#         MERCHANTBENCH_AGENT_TOKEN
#      and OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME via --env-file.
#      The SDK reads MERCHANTBENCH_AGENT_TOKEN automatically. Your code must read
#      MERCHANTBENCH_RUN_ID and exit cleanly on HTTP 410.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /agent

# Build context is agent/, so paths below are relative to it.
COPY requirements.txt /agent/requirements.txt
RUN pip install --no-cache-dir -r /agent/requirements.txt

COPY sdk/                       /agent/sdk/
COPY submission_template/my_agent.py /agent/my_agent.py

ENTRYPOINT ["python", "/agent/my_agent.py"]
