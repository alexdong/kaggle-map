@README.md
@Python.md
@Makefile

When you are asked to plan, produce a unique PLAN-{descriptive-slug}.md file in `plans` folder. 
Make sure you include the design decisions, sections and use `[ ]` in front of the TODO items.
It's often a good idea to list files you plan to change or create, key data structures involved
and the signatures of key functions. Review @.claude/agents/datastructure.md for guidance.
When you finish the planning, use osx's `say` command to announce "Plan is ready for review".

When you start making modifications, it's important to do so incrementally and test your changes thoroughly.
Review @.claude/agents/test.md for guidance. Whenever possible, create a failing test before implementing your changes.
When you finish this step, use `say` to announce "initial implementation complete, now I'll carefully 
rewrite the code according to your style guidelines".

After your code changes have passed testing, don't stop there. Follow the guidelines with one Task each:
- @.claude/agents/debuggability.md
- @.claude/agents/observability.md
- @.claude/agents/pythonic.md
- @.claude/agents/readability.md
- @.claude/agents/technicalwriter.md

After incorporating all the feedback from the previous Tasks, always run `make dev`. 
If there are outstanding issues, use `say` to announce "There are outstanding issues from linting and type checker. 
I'll carefully consider each issue and ultrathink about how I can fix them properly." 
Then go ahead and fix the issues. DO NOT attempt to add `# noqa`, `ignore` or even modify the `pyproject.toml` file.

Always run `make test` after you've made changes. 