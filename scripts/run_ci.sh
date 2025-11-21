#!/bin/bash
# CI Simulation Script - Runs all GitHub CI checks locally

set -e  # Exit on first error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track overall status
FAILED=0

echo "=========================================="
echo "  TrainCW CI Simulation"
echo "=========================================="
echo ""

# Change to project root
cd "$(dirname "$0")/.."

# ==========================================
# JOB 1: Lint & Format
# ==========================================
echo -e "${BLUE}[1/4] Lint & Format${NC}"
echo "--------------------------------------"

echo -n "Installing linting dependencies... "
pip install -q ruff mypy
echo -e "${GREEN}✓${NC}"

echo -n "Running ruff linter... "
if ruff check . > /tmp/ruff_check.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    cat /tmp/ruff_check.log
    FAILED=1
fi

echo -n "Running ruff formatter check... "
if ruff format --check . > /tmp/ruff_format.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    cat /tmp/ruff_format.log
    FAILED=1
fi

echo -n "Running mypy type checker... "
if mypy src/traincw --ignore-missing-imports > /tmp/mypy.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${YELLOW}⚠ WARNINGS (continuing)${NC}"
    cat /tmp/mypy.log
fi

echo ""

# ==========================================
# JOB 2: Test
# ==========================================
echo -e "${BLUE}[2/4] Test Suite${NC}"
echo "--------------------------------------"

echo "Installing package with dependencies..."
if pip install -e ".[dev]" > /tmp/install.log 2>&1; then
    echo -e "${GREEN}✓ Package installed${NC}"
else
    echo -e "${RED}✗ Installation failed${NC}"
    tail -50 /tmp/install.log
    FAILED=1
    exit 1
fi

echo "Running pytest..."
if python -m pytest -v --cov=traincw --cov-report=term-missing --cov-report=xml 2>&1 | tee /tmp/pytest.log; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    FAILED=1
fi

echo ""

# ==========================================
# JOB 3: Build Distribution
# ==========================================
echo -e "${BLUE}[3/4] Build Distribution${NC}"
echo "--------------------------------------"

echo -n "Installing build tools... "
pip install -q build twine
echo -e "${GREEN}✓${NC}"

echo -n "Building distribution... "
if python -m build > /tmp/build.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    cat /tmp/build.log
    FAILED=1
fi

echo -n "Checking distribution... "
if twine check dist/* > /tmp/twine.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    cat /tmp/twine.log
    FAILED=1
fi

echo ""

# ==========================================
# JOB 4: Installation Test
# ==========================================
echo -e "${BLUE}[4/4] Installation Test${NC}"
echo "--------------------------------------"

echo -n "Testing CLI... "
if traincw --version > /tmp/cli_version.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    cat /tmp/cli_version.log
    FAILED=1
fi

echo -n "Testing CLI help... "
if traincw --help > /tmp/cli_help.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    cat /tmp/cli_help.log
    FAILED=1
fi

echo -n "Testing import... "
if python -c "import traincw; print(f'TrainCW {traincw.__version__} imported successfully')" > /tmp/import.log 2>&1; then
    echo -e "${GREEN}✓ PASS${NC}"
    cat /tmp/import.log
else
    echo -e "${RED}✗ FAIL${NC}"
    cat /tmp/import.log
    FAILED=1
fi

echo ""
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All CI checks PASSED! ✅${NC}"
    echo "=========================================="
    exit 0
else
    echo -e "${RED}Some CI checks FAILED ❌${NC}"
    echo "=========================================="
    exit 1
fi
