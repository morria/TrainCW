---
description: Run ruff linter and formatter checks
---

Check code quality using ruff:

1. Run `ruff check .` to identify linting issues
2. Run `ruff format --check .` to check formatting
3. Report any issues found with file paths and line numbers
4. If issues are found, ask if I should auto-fix them with `ruff check --fix .` and `ruff format .`
5. Run mypy type checking with `mypy src/traincw` for type safety

Provide a summary of code quality status.
