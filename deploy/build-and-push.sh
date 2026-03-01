#!/bin/bash
# ============================================================
# WorldLens — Docker Build & Push Script
# ============================================================
# Usage:
#   ./deploy/build-and-push.sh                    # Build only
#   ./deploy/build-and-push.sh --push             # Build + push to Docker Hub
#   ./deploy/build-and-push.sh --push --tag v1.1  # Build + push with custom tag
# ============================================================

set -euo pipefail

# Defaults
DOCKER_USER="${DOCKER_USER:-}"
IMAGE_NAME="worldlens"
TAG="${TAG:-latest}"
PUSH=false

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --push)   PUSH=true; shift ;;
    --tag)    TAG="$2"; shift 2 ;;
    --user)   DOCKER_USER="$2"; shift 2 ;;
    *)        echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Ensure we're in the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  WorldLens Docker Build"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Project root: $PROJECT_ROOT"
echo "  Image:        ${DOCKER_USER:+$DOCKER_USER/}$IMAGE_NAME:$TAG"
echo "  Push:         $PUSH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build
FULL_IMAGE="${DOCKER_USER:+$DOCKER_USER/}$IMAGE_NAME:$TAG"
echo ""
echo "▶ Building Docker image..."
docker build \
  -t "$FULL_IMAGE" \
  -f deploy/Dockerfile \
  .

echo ""
echo "✅ Image built: $FULL_IMAGE"

# Push (if requested)
if [ "$PUSH" = true ]; then
  if [ -z "$DOCKER_USER" ]; then
    echo ""
    echo "❌ --push requires DOCKER_USER env var or --user flag"
    echo "   Example: DOCKER_USER=yourusername ./deploy/build-and-push.sh --push"
    exit 1
  fi

  echo ""
  echo "▶ Pushing to Docker Hub..."
  docker push "$FULL_IMAGE"

  # Also tag as latest if not already
  if [ "$TAG" != "latest" ]; then
    docker tag "$FULL_IMAGE" "$DOCKER_USER/$IMAGE_NAME:latest"
    docker push "$DOCKER_USER/$IMAGE_NAME:latest"
    echo "✅ Also pushed: $DOCKER_USER/$IMAGE_NAME:latest"
  fi

  echo ""
  echo "✅ Pushed: $FULL_IMAGE"
  echo ""
  echo "Others can now run:"
  echo "  docker pull $FULL_IMAGE"
  echo "  docker run -p 8000:8000 --env-file .env $FULL_IMAGE"
fi

echo ""
echo "Done!"
