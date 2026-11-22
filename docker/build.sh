#!/usr/bin/env bash
# Build both images for the father's PC (linux/amd64) and save them into one file to copy over.
set -euo pipefail
cd "$(dirname "$0")"
docker buildx build --platform linux/amd64 -t chesspgn-app:latest -f Dockerfile.app --load ..
docker buildx build --platform linux/amd64 -t chesspgn-ollama:latest -f Dockerfile.ollama --load .
mkdir -p ../dist
docker save chesspgn-app:latest chesspgn-ollama:latest -o ../dist/chesspgn-images.tar
cp docker-compose.yml "Start Chess PGN.bat" "Install Chess PGN.bat" ../dist/
echo "Done: copy the dist/ folder to the PC."
