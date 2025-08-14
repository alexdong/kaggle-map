# Python Template Documentation

Welcome to the Python Template documentation. This is a modern Python project template with comprehensive tooling and best practices built-in.

## 🚀 Quick Start

1. Clone the repository: `git clone https://github.com/username/python-template.git`
2. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. Install dependencies: `uv pip install -r pyproject.toml`
4. Run tests: `make test`
5. Start developing!

## ✨ Features

- **Modern Tooling**: Pre-configured with uv, ruff, pytest, and pydantic
- **Best Practices**: Follows Python best practices with comprehensive coding standards
- **GitHub Pages Ready**: This documentation is hosted on GitHub Pages
- **AI-Friendly**: Includes CLAUDE.md and llms/ directory with AI-optimized documentation

## 📁 Project Structure

```
./
├── docs/           # Documentation (GitHub Pages)
├── src/            # Application code
├── tests/          # Test files
├── logs/           # Implementation logs
├── llms/           # LLM-friendly documentation
├── Makefile        # Task automation
├── pyproject.toml  # Project configuration
├── Python.md       # Coding standards
└── CLAUDE.md       # AI instructions
```

## 🛠️ Available Commands

```bash
make dev            # Run linting and type checking
make test           # Run tests
make test-coverage  # Run tests with coverage
make type-coverage  # Check type annotation coverage
```

## 📚 Documentation

The `llms/` directory contains comprehensive documentation for common Python tools:

- **click** - Command Line Interface creation
- **numpy** - Scientific computing
- **pydantic** - Data validation and AI agents
- **rich** - Terminal formatting and styling
- **ruff** - Fast Python linter and formatter
- And many more...

## 🔧 Configuration

The project uses `pyproject.toml` for all tool configuration:
- Project metadata and dependencies
- Ruff linting and formatting rules
- Testing configuration
- Type checking settings

## 🤝 Contributing

Contributions are welcome! Please read the coding standards in `Python.md` before submitting pull requests.

---

[GitHub Repository](https://github.com/username/python-template) | 
[Issues](https://github.com/username/python-template/issues) | 
[Pull Requests](https://github.com/username/python-template/pulls)