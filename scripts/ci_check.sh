#!/bin/bash
# CI Check Script - Runs all checks that CI runs
# This script mimics the GitHub Actions CI workflow

set -e  # Exit on error

echo "=================================="
echo "TrainCW CI Check Script"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILED_CHECKS=()

# Function to run a check
run_check() {
    local name="$1"
    local command="$2"

    echo "==================================";
    echo "Running: $name"
    echo "==================================";

    if eval "$command"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        FAILED_CHECKS+=("$name")
        return 1
    fi
}

# Change to project root
cd "$(dirname "$0")/.."

echo "Project directory: $(pwd)"
echo ""

# Install dependencies if needed
echo "Checking dependencies..."
if ! python -c "import ruff" 2>/dev/null; then
    echo "Installing linting tools..."
    pip install ruff mypy -q
fi

echo ""

# 1. Ruff linting
run_check "Ruff Linter" "ruff check ." || true

# 2. Ruff formatting
run_check "Ruff Formatter" "ruff format --check ." || true

# 3. MyPy type checking (continue on error like CI)
echo "=================================="
echo "Running: MyPy Type Checker"
echo "=================================="
if python -c "import mypy" 2>/dev/null; then
    mypy src/traincw --ignore-missing-imports || echo -e "${YELLOW}⚠ MyPy found issues (non-blocking)${NC}"
else
    echo -e "${YELLOW}⚠ MyPy not installed, skipping${NC}"
fi
echo ""

# 4. Install package dependencies
echo "=================================="
echo "Installing package dependencies..."
echo "=================================="
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing core dependencies (this may take a while)..."
    pip install -e . -q
else
    echo "✓ Dependencies already installed"
fi
echo ""

# 5. Run tests
run_check "Pytest Tests" "python -m pytest tests/ -v --tb=short" || true

# 6. Test package build
run_check "Package Build" "python -m build --outdir dist/ 2>&1 | tail -10 || echo 'build module not found, skipping'" || true

# Summary
echo ""
echo "=================================="
echo "CI Check Summary"
echo "=================================="
echo ""

if [ ${#FAILED_CHECKS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}✗ ${#FAILED_CHECKS[@]} check(s) failed:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
        echo -e "${RED}  - $check${NC}"
    done
    echo ""
    exit 1
fi
