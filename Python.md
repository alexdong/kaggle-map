# Python Coding Standards for AI Agents

## Development Environment

### Core Tools

- **Python**: 3.13+
- **Package Management**: `uv` (not pip or requirements.txt)
- **Typing**: `pydantic` (types and validation)
- **Code Quality**: `ruff` (formatting and linting), `ty` (type checking)
- **Logging**: `loguru`
- **Testing**: `pytest`
- **CLI Tools**: `click`, `typer`, `prompt-toolkit`, `rich` and `shellingham`
- **Web**: `python-fasthtml`, `requests`, and `fh-plotly`
- **Templates**: `jinja2`
- **AI/LLM**: use anthropic by default using `pydantic-ai`
- **Data & Analysis**: `numpy`, `pandas`, `plotly`, `kaleido`
- **Cross platform**: `platformdirs`
- **Hosting**: `cloudflare tunnel`

### System Environment

- **OS**: macOS 15.5+ or latest Ubuntu with dev tools
- **Shell**: zsh with oh-my-zsh
- **API Keys**: Claude and OpenAI (environment variables)

### Rust CLI Tools

- `curl` - Enhanced HTTP client
- `jq` - JSON processor

## Project Structure

```
./
├── .claude/                # Claude Code slash commands
├── docs/                   # Documentation
├── {package_name}/         # Application code
│   ├── cli/                # CLI related code
│   ├── component/          # Component code and tests
│   ├── db/                 # Database access code and tests
│   ├── models/             # Entities and models
│   └── utils/              # Utility code and tests
│   └── web/                # Web code
├── tests/
│   ├── integration/        # Integration tests
│   └── e2e/                # End-to-end tests
├── Makefile                # Task automation
└── pyproject.toml          # Project configuration
```

@.claude/agents/datastructure.md
@.claude/agents/debuggability.md
@.claude/agents/observability.md
@.claude/agents/pythonic.md
@.claude/agents/readability.md
@.claude/agents/technicalwriter.md
@.claude/agents/test.md
