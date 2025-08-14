Perform a systematic code review based on project coding standards.


**Input Requirements:**
- Target files/directories to review: $TARGET_FILES
- Path to coding standards document: $STANDARDS_DOC (default: ./Python.md)
- Review scope: modified files and other files related.

**Review Process:**

Use the following subagents in parallel to review code changes indepdently of each other.

1. debuggability
2. observability
3. pythonic
4. readability
5. test
6. typechecker
7. technicalwriter

Collect feedback from each agent and compile them into a list.

Agent: [Agent Name]
   File: [filepath]
   Line: [line_number]
   Issue: [Brief description]
   Details: [Explain why this violates the guideline and its impact]
