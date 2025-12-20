#!/usr/bin/env bash
# Railway build script

set -o errexit

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build complete!"
