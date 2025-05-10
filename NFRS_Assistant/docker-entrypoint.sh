#!/bin/bash
set -e

echo "Waiting for PostgreSQL to start..."
while ! nc -z $POSTGRES_HOST $POSTGRES_PORT; do
  sleep 0.1
done
echo "PostgreSQL started"

echo "Applying database migrations..."
if ! python manage.py migrate; then
  echo "Warning: Migrations failed, continuing..."
fi
mkdir -p /app/vector_store
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ] && [ -n "$DJANGO_SUPERUSER_EMAIL" ]; then
  echo "Creating superuser..."; python manage.py createsuperuser --noinput
fi

exec "$@"