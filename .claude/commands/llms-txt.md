---
description: Create a comprehensive llms.txt document for a Python package with deep research
allowed-tools: WebSearch, WebFetch, FileEdit
---
You need to create a comprehensive llms.txt document for the Python package: $ARGUMENTS

This is a deep research task. Follow these steps systematically:

## 0. Check for Existing llms.txt File
First, check if the package already has an llms.txt file available:
- Search for "$ARGUMENTS llms.txt"
- Try to fetch these URLs directly:
  - Official documentation site + "/llms.txt" (e.g., https://docs.$ARGUMENTS.com/llms.txt)
  - Official documentation site + "/llms-full.txt"
  - PyPI project page llms.txt if applicable
  - GitHub repo raw content llms.txt if applicable
- Check known llms.txt directories like directory.llmstxt.cloud for the package

If an llms.txt file is found:
- Fetch and review the existing file
- If it's comprehensive and up-to-date, save it as `$ARGUMENTS-llms.txt`
- Add a note at the top: "# Note: This is the official llms.txt from [source URL]"
- Inform the user that an official llms.txt was found and used
- Skip the remaining steps

If no existing llms.txt is found, proceed with creating one:

## Understanding llms.txt vs llms-full.txt
- **llms.txt**: A navigation/index file with links to resources (what we're creating)
- **llms-full.txt**: Contains all documentation content in a single file
- Some projects provide both; we're focusing on the index version
- Use web_fetch to check if the project provides either format

## 1. Research Phase (use multiple web searches)
- Search for "$ARGUMENTS official documentation"
- Search for "$ARGUMENTS tutorial getting started"
- Search for "$ARGUMENTS best practices"
- Search for "$ARGUMENTS common mistakes pitfalls"
- Search for "$ARGUMENTS code examples"
- Search for "$ARGUMENTS API reference"
- Use web_fetch to read the official documentation pages
- Look for recent blog posts and tutorials about $ARGUMENTS

## 2. Structure the llms.txt Document
Create a file named `$ARGUMENTS-llms.txt` following the official llms.txt format:

```
# $ARGUMENTS

> [Brief summary of what $ARGUMENTS is and its primary purpose - this is the key information needed to understand the rest of the file]

[Optional detailed information about the package, important notes, compatibility info, etc.]

## Getting Started
- [Installation Guide](link): How to install $ARGUMENTS
- [Quick Start](link): Minimal working example
- [Basic Tutorial](link): Introduction to core concepts

## Core Documentation  
- [API Reference](link): Complete API documentation
- [User Guide](link): Comprehensive usage guide
- [Configuration](link): Configuration options and settings

## Examples
- [Basic Examples](link): Simple usage patterns
- [Advanced Examples](link): Complex real-world examples
- [Code Samples](link): Repository of example code

## Optional
- [Contributing Guide](link): How to contribute to the project
- [Changelog](link): Recent updates and version history
- [FAQ](link): Frequently asked questions
```

## 3. Content Requirements
Since llms.txt is primarily a navigation file pointing to detailed content:
- **Format**: Follow the official llms.txt markdown format exactly
- **Links**: Each list item should be a markdown link with optional description after a colon
- **Conciseness**: Keep descriptions brief and informative
- **Organization**: Group related links under appropriate H2 headers
- **Optional Section**: Use "## Optional" for secondary resources that can be skipped
- **No Direct Content**: Don't include code examples or detailed explanations in the llms.txt itself
- **External Links**: Can include links to external resources if helpful

## 4. Creating Companion Content (if needed)
If official documentation lacks markdown versions:
- Note which key documentation pages should ideally have .md versions
- Suggest creating llms-ctx.txt or llms-full.txt for expanded context
- Mention the llms_txt2ctx tool for generating these expanded versions

## 5. Format Guidelines
- Use standard markdown formatting
- Start with H1 heading: `# PackageName`
- Follow with blockquote summary: `> Brief description...`
- Use H2 headings for sections (e.g., `## Documentation`, `## Examples`)
- Each item should be a markdown link: `- [Title](URL): Optional description`
- Keep descriptions concise and informative
- Include an `## Optional` section for secondary resources

Important:
- The llms.txt file is a curated index, not a comprehensive guide
- Focus on providing navigation to the most useful resources
- Prioritize official documentation and high-quality resources
- The file should help LLMs quickly find relevant information
- Consider creating llms-ctx.txt or llms-full.txt for expanded versions
- Follow the specification at https://llmstxt.org/

Arguments: The name of the Python package to document (e.g., plotly, pandas, fastapi)
