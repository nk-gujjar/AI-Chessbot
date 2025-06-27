# build_app.sh
#!/bin/bash

echo "Building Chess Bot Mac Application..."

# Clean previous builds
rm -rf build dist

# Create necessary directories
mkdir -p models
mkdir -p games

# Install dependencies
pip3 install -r requirements.txt

# Build the app
python3 setup.py py2app

echo "Build complete! Application is in the 'dist' folder."
echo "To run: open dist/ChessBot.app"
