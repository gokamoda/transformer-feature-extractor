

# Usage

- Install dependencies:
    ```bash
    make install
    ```
- Run the greet command:
    ```bash
    make greet
    ```

- Run greet with overrides:
    ```bash
    uv run greet --config configs/debug.yaml --override debug.message=hello
    ```