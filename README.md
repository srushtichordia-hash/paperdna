# PaperDNA

An NLP research discovery engine that maps agreement, disagreement, and condition-dependent scientific findings.

> **Status:** early foundation. Only repository architecture, configuration, and
> testing are in place. NLP models, dataset ingestion, retrieval, and the UI are
> not implemented yet.

## Requirements

- Python >= 3.10
- `pip` and `venv` (included with standard Python installations)

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/srushtichordia-hash/paperdna.git
cd paperdna
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## Installation

Install the package in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs PaperDNA with no runtime dependencies; `pytest` is the only
development dependency.

## Running the tests

```bash
pytest
```

## Configuration

PaperDNA reads configuration from environment variables with the `PAPERDNA_`
prefix and safe defaults, so no setup is required to get started. It does not
parse `.env` files itself — export the variables in your shell or use a tool of
your choice.

| Variable            | Default | Description                          |
| ------------------- | ------- | ------------------------------------ |
| `PAPERDNA_LOG_LEVEL`| `INFO`  | Logging level (`DEBUG`, `INFO`, ...) |

Example:

```bash
export PAPERDNA_LOG_LEVEL=DEBUG
```

See [`.env.example`](.env.example) for a documented template. Never commit real
secrets; `.env` files are gitignored.

## Project layout

```
src/paperdna/       Package source (config, logging)
tests/              Pytest test suite (smoke tests)
pyproject.toml      Project metadata, dependencies, and tool config
.env.example        Example environment configuration
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
