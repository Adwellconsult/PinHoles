#!/bin/bash
# Quick setup script for Bullet Impact Detector development

echo "🎯 Bullet Impact Detector - Development Setup"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source .venv/Scripts/activate
else
    # Linux/Mac
    source .venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found, installing basic dependencies..."
    pip install kivy opencv-python numpy
fi

# Install Buildozer for Android builds (optional)
read -p "🤖 Install Buildozer for Android building? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔨 Installing Buildozer..."
    pip install buildozer cython
    echo "✅ Buildozer installed"
else
    echo "⏭️ Skipping Buildozer installation"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Run desktop app: python bullet_detector_android.py"
echo "  3. Build Android APK: buildozer android debug"
echo ""
echo "📖 See README.md for detailed usage instructions"