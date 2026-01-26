FROM python:3.11-slim

# Install Node.js for building frontend
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and build frontend
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

COPY frontend ./frontend
RUN cd frontend && npm run build

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend
COPY backend ./backend

# Copy built frontend to backend static folder
RUN mkdir -p backend/static && cp -r frontend/dist/* backend/static/

WORKDIR /app/backend

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
