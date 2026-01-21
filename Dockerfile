FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /vizora

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files first (for Docker cache)
COPY pyproject.toml poetry.lock* /vizora/

# Configure Poetry to install into system site-packages
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --only main --no-root

# Copy the rest of the application
COPY . /vizora/

# Expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "vizora.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
