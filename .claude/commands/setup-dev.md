---
description: Set up development environment from scratch
---

Set up a complete development environment for TrainCW:

1. **Check Python version**
   - Verify Python 3.10+ is available
   - Show current version

2. **Create virtual environment**
   - Guide through venv or conda setup
   - Activate the environment

3. **Install dependencies**
   - Install core dependencies: `pip install -e .`
   - Install dev dependencies: `pip install -e ".[dev]"`
   - Optionally install export tools: `pip install -e ".[export]"`
   - Optionally install viz tools: `pip install -e ".[viz]"`

4. **Set up pre-commit hooks** (optional but recommended)
   - Install pre-commit: `pip install pre-commit`
   - Set up hooks: `pre-commit install`

5. **Verify installation**
   - Run `traincw --version`
   - Run `pytest --version`
   - Run `ruff --version`
   - Run a quick test: `pytest tests/ -v`

6. **Show next steps**
   - Point to DESIGN.md for architecture overview
   - Suggest running `/check-all` to verify setup

Provide clear, step-by-step instructions.
