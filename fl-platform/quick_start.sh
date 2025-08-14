#!/bin/bash

# FL Platform Quick Start Script
# Run this on your Azure VM after cloning the repository

set -e

echo "========================================="
echo "FL Platform Quick Start"
echo "========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please don't run as root/sudo"
   exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing..."
    sudo apt update
    sudo apt install -y docker.io docker-compose
    sudo usermod -aG docker $USER
    echo "Docker installed. Please logout and login again, then re-run this script."
    exit 0
fi

# Check Docker daemon
if ! docker ps &> /dev/null; then
    echo "Docker daemon not running or you need to logout/login for group changes"
    exit 1
fi

# Get public IP
echo "Enter your VM's public IP address (or press Enter for localhost):"
read PUBLIC_IP
PUBLIC_IP=${PUBLIC_IP:-localhost}

# Setup environment
if [ ! -f .env ]; then
    cp .env.example .env
    sed -i "s|PUBLIC_BASE_URL=.*|PUBLIC_BASE_URL=http://$PUBLIC_IP:8000|" .env
    echo "Created .env file with PUBLIC_BASE_URL=http://$PUBLIC_IP:8000"
fi

# Create necessary directories
mkdir -p storage logs intake

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services
echo "Waiting for services to start..."
sleep 10

# Check health
echo "Checking service health..."
curl -f http://localhost:8000/health || echo "Warning: Server not responding yet"

echo ""
echo "========================================="
echo "FL Platform is starting!"
echo "========================================="
echo ""
echo "Services:"
echo "  - API Server: http://$PUBLIC_IP:8000"
echo "  - API Docs: http://$PUBLIC_IP:8000/docs"
echo "  - Redis: localhost:6379"
echo ""
echo "Next steps:"
echo "1. Submit a task:"
echo "   curl -X POST http://$PUBLIC_IP:8000/upload \\"
echo "     -H 'X-API-Token: demo-token-123' \\"
echo "     -F 'task=@samples/task.py' \\"
echo "     -F 'config=@samples/run.yaml'"
echo ""
echo "2. Start a client (from any machine):"
echo "   export FL_SERVER_URL=http://$PUBLIC_IP:8000"
echo "   export CLIENT_ID=client-001"
echo "   export API_TOKEN=demo-token-123"
echo "   python client/fl_client.py"
echo ""
echo "3. Check logs:"
echo "   docker-compose logs -f"
echo ""
echo "4. Stop services:"
echo "   docker-compose down"
echo ""