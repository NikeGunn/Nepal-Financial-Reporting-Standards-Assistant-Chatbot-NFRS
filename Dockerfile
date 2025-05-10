FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=NFRS_Assistant.settings.prod

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for production
RUN apt-get update && apt-get install -y --no-install-recommends \
  netcat-traditional \
  && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/media /app/vector_store /app/logs /app/staticfiles

# Collect static files
RUN python manage.py collectstatic --noinput

# Copy the entrypoint script
COPY ./docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Run the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "NFRS_Assistant.wsgi:application"]