# syntax=docker/dockerfile:1.7

# ─── Stage 1: Node build ──────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /build

# Copy manifests first to leverage Docker layer caching.
# A lockfile (package-lock.json / yarn.lock) is intentionally absent from
# the source tree so we use `npm install` to generate one on first build.
COPY package.json ./

RUN npm install --frozen-lockfile 2>/dev/null || npm install

# Copy source and build for production
COPY . .

# tsc --noEmit verifies types; vite build produces the optimised dist/
RUN npm run build

# ─── Stage 2: Nginx runtime ───────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

LABEL maintainer="pwr-twin-team"
LABEL description="PWR Digital Twin — React/Vite frontend served by Nginx"

# Remove the default catch-all vhost
RUN rm -f /etc/nginx/conf.d/default.conf

# Install our custom vhost config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the production build
COPY --from=builder /build/dist /usr/share/nginx/html

# Nginx listens on 80 inside the container;
# docker-compose maps host port 3000 → container 80
EXPOSE 80

# Validate config before starting
#RUN nginx -t

CMD ["nginx", "-g", "daemon off;"]
