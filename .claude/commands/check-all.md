---
description: Run all quality checks (tests, linting, formatting, types)
---

Perform a comprehensive quality check of the codebase:

1. **Formatting Check**: Run `ruff format --check .`
2. **Linting**: Run `ruff check .`
3. **Type Checking**: Run `mypy src/traincw`
4. **Tests**: Run `pytest -v --cov=traincw`

Provide a comprehensive summary showing:
- ✅ Checks that passed
- ❌ Checks that failed with details
- Overall project health status
- Recommendations for fixing any issues

This command is ideal before creating PRs or commits.
