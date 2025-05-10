FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=NFRS_Assistant.settings.prod

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y --no-install-recommends \
  netcat-traditional \
  libmagic1 \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/media /app/vector_store /app/logs /app/staticfiles

# Verify project structure (for debugging)
RUN ls -la /app/NFRS_Assistant && ls -la /app/NFRS_Assistant/settings

# Collect static files
RUN python manage.py collectstatic --noinput

# Copy and set permissions for the entrypoint script
COPY ./docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Run the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "NFRS_Assistant.wsgi:application"]