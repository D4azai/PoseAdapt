FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/PoseAdapt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN python -m pip install --upgrade pip && python -m pip install -e .

ENTRYPOINT ["python", "scripts/infer_image.py"]
CMD ["--help"]
