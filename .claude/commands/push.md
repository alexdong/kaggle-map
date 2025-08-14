---
description: Run development checks, commit all changes, and push to GitHub
allowed-tools: Bash
---

You need to commit and push all changes to GitHub following the project workflow.

Execute these steps in sequence:

1. **Run development checks**: Execute `make dev` to format, lint, and type-check the code
2. **Run tests**: Execute `make test` to ensure all tests pass
3. **Check git status**: Run `git status` to see what changes are staged/unstaged
4. **Add all changes**: Run `git add .` to stage all changes
5. **Commit with message**: Create a commit with a meaningful message based on the changes
6. **Push to origin**: Run `git push origin main` to push to GitHub

Important: 
- If `make dev` or `make test` fail, fix the issues before committing
- Write a clear commit message that describes what was changed and why

Arguments (optional): $ARGUMENTS can be used as the commit message if provided, otherwise analyze the changes to create an appropriate message.