# Contributing to AquaCal

Thank you for your interest in contributing to AquaCal. This guide will help you get started with development.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tlancaster6/AquaCal.git
   cd AquaCal
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests to verify your setup:
   ```bash
   python -m pytest tests/
   ```

## Code Style

- **Formatter**: Black (run `black src/ tests/` before committing)
- **Docstrings**: Google style
- **Type hints**: Use `numpy.typing.NDArray` with shape information in docstrings
- **Imports ordering**:
  1. Standard library imports
  2. Third-party imports (numpy, scipy, opencv, etc.)
  3. Local imports (from aquacal)

  Separate each group with a blank line.

## Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Skip slow optimization tests:
```bash
python -m pytest tests/ -m "not slow"
```

Run a single test file:
```bash
python -m pytest tests/unit/test_camera.py -v
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and add tests
4. Ensure all tests pass
5. Format your code with Black
6. Commit your changes with a clear message
7. Push to your fork and submit a pull request

## Deprecation Policy

When deprecating functionality in AquaCal:

1. **Add a deprecation warning** using `warnings.warn()` with `DeprecationWarning`:
   ```python
   import warnings

   def old_function():
       warnings.warn(
           "old_function() is deprecated as of version 1.2.0 and will be removed in version 1.4.0. "
           "Use new_function() instead.",
           DeprecationWarning,
           stacklevel=2
       )
       # existing implementation
   ```

2. **Document in CHANGELOG.md** under the "Deprecated" category:
   ```markdown
   ### Deprecated
   - `old_function()` - Use `new_function()` instead (will be removed in 1.4.0)
   ```

3. **Maintain for at least 2 minor versions** before removal.

4. **Document the replacement** in the function's docstring:
   ```python
   def old_function():
       """Original description.

       .. deprecated:: 1.2.0
           Use :func:`new_function` instead. Will be removed in version 1.4.0.
       """
   ```

Always include `stacklevel=2` in warnings to show the caller's location, not the warning itself.
