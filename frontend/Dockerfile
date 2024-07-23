# Build stage
FROM node:14 as build

WORKDIR /app

COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

# Add a script to replace the API_URL at runtime
RUN echo '#!/bin/sh' > /docker-entrypoint.d/40-replace-api-url.sh && \
    echo 'sed -i "s|https://backend-api-gztrxlc7wa-nw.a.run.app|$API_URL|g" /usr/share/nginx/html/index.html' >> /docker-entrypoint.d/40-replace-api-url.sh && \
    chmod +x /docker-entrypoint.d/40-replace-api-url.sh

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]