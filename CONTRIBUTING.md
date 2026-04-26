# Contributing to VLA-Arena

Thank you for your interest in contributing to VLA-Arena! We welcome contributions from the community to help improve and expand this benchmark for Vision-Language-Action models.

## How to Contribute

If you are interested in contributing to VLA-Arena, your contributions will fall into two categories:

1. **You want to propose a new Feature and implement it**
    - Create an issue about your intended feature, and we shall discuss the design and implementation. Once we agree that the plan looks good, go ahead and implement it.
    - For new tasks or environments, please ensure they follow the existing structure and provide appropriate documentation and examples.

2. **You want to implement a feature or bug-fix for an outstanding issue**
    - Look at the outstanding issues here: <https://github.com/PKU-Alignment/VLA-Arena/issues>
    - Pick an issue or feature and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to <https://github.com/PKU-Alignment/VLA-Arena>

If you are not familiar with creating a Pull Request, here are some guides:

- <http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request>
- <https://help.github.com/articles/creating-a-pull-request/>

## Development Setup

To develop VLA-Arena on your machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/PKU-Alignment/VLA-Arena
cd VLA-Arena
```

### 2. Install in Development Mode

Install VLA-Arena in editable mode with all development dependencies:

```bash
pip install -e .
pip install -e '.[lint,test,docs]'
```

Or use the Makefile:

```bash
make install-editable
```

### 3. Install Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
make pre-commit-install
```

This will automatically run linters and formatters before each commit.

## Code Style

We follow strict code style guidelines to maintain consistency across the codebase:

### Formatting

- We use [black](https://github.com/psf/black) for code formatting (line length of 100 characters)
- We use [isort](https://github.com/timothycrosley/isort) to sort imports
- We use [ruff](https://github.com/astral-sh/ruff) for fast Python linting

**Please run `make format`** to automatically format your code before committing.

### Linting

You can check your code style and quality using:

```bash
make lint
```

This will run:
- `ruff` - Fast Python linter
- `flake8` - Style guide enforcement
- `pylint` - Static code analysis
- `black --check` - Code formatting check
- `isort --check` - Import sorting check

### Type Annotations

Please document each function/method and add type annotations using the following template:

```python
def my_function(arg1: type1, arg2: type2) -> returntype:
    """Short description of the function.

    Args:
        arg1: Describe what is arg1
        arg2: Describe what is arg2

    Returns:
        Describe what is returned
    """
    ...
    return my_variable
```

We follow Google-style docstrings. For more information, see:
- <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>

## Testing

All new features must add tests in the `tests/` folder ensuring that everything works fine. We use [pytest](https://pytest.org/).

Also, when a bug fix is proposed, tests should be added to avoid regression.

### Running Tests

To run tests with `pytest`:

```bash
make test
```

Or directly with pytest:

```bash
pytest tests/ -v --cov=vla_arena
```

### Test Coverage

We aim for high test coverage. Please ensure your changes include appropriate tests. You can check coverage with:

```bash
make pytest
```

This will generate a coverage report showing which lines are covered by tests.

## Pull Request (PR) Guidelines

Before proposing a PR, please follow these guidelines:

1. **Open an Issue First**: Discuss the feature or bug fix in an issue before implementing it. This prevents duplicated effort and ensures alignment with project goals.

2. **Code Quality**: Ensure your code passes all linting and formatting checks:
   ```bash
   make lint
   make format
   ```

3. **Tests**: Add appropriate tests for your changes:
   ```bash
   make test
   ```

4. **Documentation**: Update documentation if needed, including:
   - Code docstrings
   - README updates
   - New example scripts if applicable

5. **Commit Messages**: Use clear, descriptive commit messages following conventional commits format:
   - `feat: Add new task for object manipulation`
   - `fix: Correct reward calculation in evaluation`
   - `docs: Update installation instructions`
   - `refactor: Simplify policy interface`
   - `test: Add tests for new environment`

6. **PR Description**: Provide a clear description of:
   - What the PR does
   - Why the change is needed
   - How to test the changes
   - Any breaking changes or migration notes

7. **Review Process**: Each PR must be reviewed and accepted by at least one maintainer. A PR must pass all Continuous Integration (CI) tests to be merged.

## Development Commands

Here are useful commands for development:

```bash
# Install in editable mode
make install-editable

# Format code
make format

# Run linters
make lint

# Run tests
make test

# Run all checks before committing
make commit-checks

# Install pre-commit hooks
make pre-commit-install

# Clean build artifacts
make clean
```

## Adding New Tasks

When adding new tasks to VLA-Arena:

1. Create task definition in `vla_arena/vla_arena/envs/`
2. Add BDDL files in `vla_arena/vla_arena/bddl_files/`
3. Add task configuration in `vla_arena/configs/task_suite/`
4. Update documentation with task description and examples
5. Add tests for the new task
6. Include example demonstration data if possible

## Adding New Evaluation Metrics

When adding new evaluation metrics:

1. Implement metric in `vla_arena/evaluation/`
2. Add metric to evaluator configuration
3. Update documentation with metric description
4. Add tests for the metric
5. Provide examples of metric usage

## Documentation

Documentation is crucial for users and contributors. When making changes:

1. Update docstrings for any modified functions or classes
2. Update README.md if adding new features or changing installation
3. Add examples in `scripts/` if adding new functionality
4. Keep documentation clear and concise

This will build and serve the documentation locally.

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Open a new issue with the `question` label
3. Reach out to the maintainers


Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.
