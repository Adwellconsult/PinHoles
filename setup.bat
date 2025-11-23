@echo off
REM Quick setup script for Bullet Impact Detector development (Windows)

echo 🎯 Bullet Impact Detector - Development Setup
echo ==============================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3 is required but not installed
    echo Please install Python 3.8+ from python.org and try again
    pause
    exit /b 1
)

echo ✅ Python found:
python --version

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing Python dependencies...
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo ⚠️ requirements.txt not found, installing basic dependencies...
    pip install kivy opencv-python numpy
)

REM Install Buildozer for Android builds (optional)
set /p INSTALL_BUILDOZER="🤖 Install Buildozer for Android building? (y/n): "
if /i "%INSTALL_BUILDOZER%"=="y" (
    echo 🔨 Installing Buildozer...
    pip install buildozer cython
    echo ✅ Buildozer installed
) else (
    echo ⏭️ Skipping Buildozer installation
)

echo.
echo 🎉 Setup complete!
echo.
echo 🚀 Next steps:
echo   1. Activate environment: .venv\Scripts\activate.bat
echo   2. Run desktop app: python bullet_detector_android.py
echo   3. Build Android APK: buildozer android debug
echo.
echo 📖 See README.md for detailed usage instructions

pause