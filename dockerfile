FROM ubuntu:25.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y python3-pip

# TODO: Our Python is limited below 3.13, because SentencePiece problems. Fix another day
RUN apt-get install -y python3.12

RUN pip install poetry --break-system-packages

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock README.md ./

RUN poetry env use python3.12
RUN poetry install --no-root

COPY /src/. /app/src
COPY /scripts/. /app/scripts

ENV PYTHONPATH="${PYTHONPATH}:/app" \
    VIRTUAL_ENV=/app/.venv \
    PATH="$VIRTUAL_ENV/bin:/root/.local/bin:$PATH"
