# ── The database: PostgreSQL 18, from its official image ─────────────────────

FROM postgres:18-alpine

# Start through setup.sh, which gives PostgreSQL its settings from .env
COPY --chmod=755 setup/setup.sh /usr/local/bin/
ENTRYPOINT ["setup.sh"]
CMD ["postgres"]
