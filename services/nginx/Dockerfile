# ── The reverse proxy: nginx, the only container with a port open on the computer ──

FROM debian:bookworm-slim

# Install nginx, and openssl to make the certificate
RUN apt-get update \
 && apt-get install -y --no-install-recommends nginx openssl \
 && rm -rf /var/lib/apt/lists/*

# The server's certificate and private key, if certs/ holds them (apnea.crt and apnea.key)
COPY certs/ /etc/nginx/ssl/

# Without them, make a self-signed certificate for localhost, valid for 365 days from the build
RUN if [ ! -f /etc/nginx/ssl/apnea.crt ] && [ ! -f /etc/nginx/ssl/apnea.key ]; then \
      openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
        -keyout /etc/nginx/ssl/apnea.key -out /etc/nginx/ssl/apnea.crt \
        -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"; \
    fi \
 && chmod 600 /etc/nginx/ssl/apnea.key

# nginx's settings: HTTPS on port 443, every request passed on to the inference service
COPY conf/nginx.conf /etc/nginx/nginx.conf

# Check the settings; fails if the certificate or the key is missing, or if they do not match
RUN nginx -t

# Listen on port 443
EXPOSE 443

# Run nginx in the foreground, as the container's main process
CMD ["nginx", "-g", "daemon off;"]
