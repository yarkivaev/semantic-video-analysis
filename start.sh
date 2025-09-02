#!/bin/sh
# Install boto3 if not present (needed for volume mounts during development)
pip list | grep -q boto3 || pip install boto3

# Initialize models (downloads on first run, uses cache on subsequent runs)
echo "Checking and initializing AI models..."
python /app/init_models.py

# Check if model initialization succeeded
if [ $? -ne 0 ]; then
    echo "Error: Failed to initialize models"
    exit 1
fi

# Start the MCP server
echo "Starting MCP server..."
exec python /app/mcp_interface/http_server.py