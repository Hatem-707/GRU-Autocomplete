#!/bin/bash
# Deployment Checklist for Autocomplete Web Frontend

echo "🚀 Next Word Autocomplete - Deployment Checklist"
echo "=================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (MISSING)"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${RED}✗${NC} $1/ (MISSING)"
        return 1
    fi
}

echo "📂 Checking Web Frontend Files..."
echo ""

FRONTEND_DIR="$(dirname "$0")"
cd "$FRONTEND_DIR"

# Check essential files
echo "Essential Files:"
check_file "index.html"
check_file "app.js"
check_file "server.py"
check_file "prepare_assets.py"

echo ""
echo "Documentation:"
check_file "README.md"
check_file "QUICKSTART.md"
check_file "ARCHITECTURE.md"
check_file "SETUP.html"

echo ""
echo "Configuration:"
check_file "config.js"

echo ""
echo "📦 Checking Generated Assets..."
echo ""

PARENT_DIR="$(dirname "$FRONTEND_DIR")"
echo "Source Files (should exist before prepare_assets.py):"
check_file "$PARENT_DIR/word2id_daily.json"
check_file "$PARENT_DIR/id2word_daily.json"
check_file "$PARENT_DIR/model_daily_gru_attention.onnx"

echo ""
echo "Generated Files (created by prepare_assets.py):"
check_file "vocabulary.json"
check_file "model.onnx"

echo ""
echo "🎯 Setup Checklist"
echo ""

PASSED=0
FAILED=0

# Test Python availability
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} Python 3 installed"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Python 3 not found"
    ((FAILED++))
fi

# Test if we can run the prepare script
if [ -f "$FRONTEND_DIR/prepare_assets.py" ]; then
    echo -e "${GREEN}✓${NC} prepare_assets.py exists and is ready to run"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} prepare_assets.py not found"
    ((FAILED++))
fi

# Test server.py
if [ -f "$FRONTEND_DIR/server.py" ]; then
    echo -e "${GREEN}✓${NC} server.py exists and is ready to run"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} server.py not found"
    ((FAILED++))
fi

echo ""
echo "📋 Next Steps"
echo ""

if [ ! -f "vocabulary.json" ] || [ ! -f "model.onnx" ]; then
    echo -e "${YELLOW}1. Prepare web assets:${NC}"
    echo "   cd '$FRONTEND_DIR'"
    echo "   python3 prepare_assets.py"
    echo ""
fi

echo -e "${YELLOW}2. Start the development server:${NC}"
echo "   cd '$FRONTEND_DIR'"
echo "   python3 server.py"
echo ""

echo -e "${YELLOW}3. Open in browser:${NC}"
echo "   http://localhost:8000"
echo ""

echo "📚 Documentation"
echo ""
echo "- QUICKSTART.md     - Fast setup guide (start here)"
echo "- README.md         - Full documentation"
echo "- ARCHITECTURE.md   - Technical details"
echo "- SETUP.html        - Visual setup guide (open in browser)"
echo ""

echo "🎉 Summary"
echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All checks passed! You're ready to go.${NC}"
    echo ""
    echo "Run: python3 prepare_assets.py"
    echo "Then: python3 server.py"
else
    echo -e "${RED}$FAILED issue(s) need attention.${NC}"
    echo ""
    echo "Make sure all source files are in: $PARENT_DIR"
    echo "- word2id_daily.json"
    echo "- id2word_daily.json"
    echo "- model_daily_gru_attention.onnx"
fi

echo ""
echo "=================================================="
