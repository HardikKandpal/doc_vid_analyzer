FROM node:16-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY . .
RUN npm run build

# Install serve to run the application
RUN npm install -g serve

EXPOSE 3000

# Use serve to run the built application
CMD ["serve", "-s", "build", "-l", "3000"]