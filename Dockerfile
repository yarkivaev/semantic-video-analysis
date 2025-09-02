# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8090

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy setup.py and install dependencies
COPY setup.py README.md /app/
COPY semantic_video_analysis /app/semantic_video_analysis/
COPY mcp_interface /app/mcp_interface/

# Install Python dependencies including HTTP extras (boto3 is included in http extras)
RUN pip install --no-cache-dir -e ".[mcp,http]"

# Expose the MCP server port
EXPOSE 8090

# Copy the model initialization script
COPY init_models.py /app/init_models.py

# Health check with longer start period for model downloads
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:8090/health || exit 1

# Copy startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Run the HTTP MCP server via startup script
CMD ["/app/start.sh"]