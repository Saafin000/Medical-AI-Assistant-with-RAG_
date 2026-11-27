#!/bin/bash
# Clinical Pre-Consultation Agent - Demonstration Script
# Run this to verify all fixes and start the application

echo "========================================================================"
echo "Clinical Pre-Consultation Agent - Demonstration Setup"
echo "========================================================================"
echo ""

# Change to project directory
cd "$(dirname "$0")" || exit 1

echo "[1/3] Running Critical Fixes Test Suite..."
echo "----------------------------------------------------------------------"
python test_critical_fixes.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tests failed! Please check the output above."
    exit 1
fi

echo ""
echo "✅ Critical tests passed!"
echo ""

echo "[2/3] Running Integration Tests..."
echo "----------------------------------------------------------------------"
python test_integration_clinical.py

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  Some integration tests failed (may be expected)"
    echo "Continue? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "✅ Tests completed!"
echo ""

echo "[3/3] Starting Application..."
echo "----------------------------------------------------------------------"
echo ""
echo "Application will start on http://localhost:8000"
echo ""
echo "Instructions:"
echo "1. Open your browser to http://localhost:8000"
echo "2. Select 'Clinical Pre-Consultation Agent'"
echo "3. Follow the demonstration conversation in DEMONSTRATION_GUIDE.md"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
