@README.md
@Python.md
@Makefile

### Development Principles

Please carefully consider the following principles. They are very different from mainstream software development practices but they are CRITICAL in long term maintainability of the code base. Follow them diligently.

#### 1. **Fail Early and Noisily**
- Use `assert` statements liberally - never try/except or throw
- Instead of comments, use detailed `loguru` logging to capture flow and variable states.

#### 2. **Can you make it simpler?**
- After your generate code changes, always rewrite it a few times until you can't make it simpler.
- Ultrathink about the types and how they flow through functions. 
- Defensive programming is the main reason for code bloat. Avoid `| None` whenever possible.

#### 3. **Test and Type Safety**
- Use Python 3.13+ features. Avoid deprecated features.
- 100% type annotations with Pydantic. Parse, don't validate. Let Pydantic handles validation at boundaries
- Write tests before code. Write tests before fixing bugs. Before working on any task, start with a test.