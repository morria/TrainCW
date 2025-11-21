# Claude Code Configuration for TrainCW

This directory contains Claude Code configuration to help with development.

## Custom Slash Commands

Use these slash commands for common tasks:

- `/test` - Run pytest with coverage reporting
- `/lint` - Run ruff linter and formatter checks
- `/check-all` - Run all quality checks (tests, linting, formatting, types)
- `/review-architecture` - Review neural network architecture implementation
- `/review-data-generation` - Review synthetic data generation implementation
- `/optimize` - Analyze code for performance optimizations
- `/add-module` - Create a new module with proper structure and tests
- `/setup-dev` - Set up development environment from scratch
- `/explain-codebase` - Get a comprehensive overview of the codebase

## Project Context

When working on this project, Claude is configured to:

- Follow PEP 8 and modern Python best practices
- Use type hints and comprehensive docstrings
- Write tests for all new functionality
- Maintain consistency with the architecture defined in DESIGN.md
- Use ruff for linting and formatting
- Run pytest for testing
- Use mypy for type checking

## Code Style

- Max line length: 100 characters
- Quote style: Double quotes
- Indentation: 4 spaces
- Import organization: isort via ruff
- Type hints: Required for all function signatures

## Development Workflow

1. Create a new branch for your feature
2. Write code following project standards
3. Run `/check-all` to verify quality
4. Write tests for new functionality
5. Run `/test` to ensure all tests pass
6. Commit with a descriptive message
7. Create a PR

## Specialized Reviews

For specific aspects of the codebase:

- **Neural network code**: Use `/review-architecture` to check model implementations
- **Data generation**: Use `/review-data-generation` to validate synthesis pipeline
- **Performance**: Use `/optimize` to identify bottlenecks and improvements

## Tips for Claude

- Always check DESIGN.md for architectural decisions
- Follow the project structure outlined in the design document
- Use the custom commands for faster development
- Ask clarifying questions when requirements are unclear
- Provide detailed explanations for complex algorithms
- Consider edge cases and error handling
- Think about maintainability and readability
