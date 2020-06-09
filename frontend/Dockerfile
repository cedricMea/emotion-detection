FROM node:alpine

WORKDIR '/app'

COPY package.json ./
RUN npm install

COPY ./ ./
RUN npm run build


# production environment
FROM nginx:stable-alpine
COPY --from=0 /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
