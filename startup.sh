#!/bin/bash
# startup.sh — Birdie Azure App Service startup script
#
# This file is NOT used as the primary startup mechanism.
# The App Service startup command is configured directly in the Azure Portal:
#
#   gunicorn -w 1 -k uvicorn.workers.UvicornWorker \
#            --bind=0.0.0.0:$PORT backend.app:app
#
# When Oryx builds the application (SCM_DO_BUILD_DURING_DEPLOYMENT=true),
# it activates its own virtualenv before running any startup command.
# You do NOT need to reference antenv or activate any venv manually.
#
# This file is kept only as documentation and a local development helper.
# It is excluded from the deployment ZIP automatically if not needed.

set -e

echo "Birdie startup — environment info:"
echo "  Python  : $(python --version 2>&1)"
echo "  PORT    : ${PORT:-8000}"
echo "  PWD     : $(pwd)"

exec gunicorn \
  -w 1 \
  -k uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT:-8000}" \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  backend.app:app
