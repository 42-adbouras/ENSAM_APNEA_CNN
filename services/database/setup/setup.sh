#!/bin/sh
# Starts PostgreSQL with the project's settings: DB_USER, DB_PASS and DB_NAME from .env, under the
# names PostgreSQL's image reads. On the first start, with no database yet, the image's own script
# uses them to create that user and database; later starts reuse what is there.
set -eu
export POSTGRES_USER="$DB_USER" POSTGRES_PASSWORD="$DB_PASS" POSTGRES_DB="$DB_NAME"
exec docker-entrypoint.sh "$@"
