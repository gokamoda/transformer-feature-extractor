# feature-extractor

## Environment setup

- Install dependencies with uv:
  ```bash
  make install
  ```
- For Codex or other local automation tools, load the project environment automatically:
  ```bash
  direnv allow
  ```
- Run linting and formatting:
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- Run tests:
  ```bash
  uv run pytest
  ```
