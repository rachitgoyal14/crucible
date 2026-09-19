## Overview

It's time to propose and implement your project. A few guidelines on selecting a good problem statement:

- Don't pick something too ambitious (e.g. creating a new operating system).
- Don't pick something too simplistic (e.g. printing "Hello World").
- Aim for something you don't yet know how to do. Example ideas:
  - Creating a simple memory allocator
  - Rewriting shell commands such as `cat`, `chmod`, `echo`
  - Writing a simple compiler
  - Writing a simple scheduler
  - Writing your own standard library
  - Making a game in C from scratch
  - Implementing solitaire
  - Implementing poker
  - Building your own shell
  - A regular expression matcher
  - Creating your own text editor
  - Writing your own libraries (encoding/decoding, sorting, searching, etc.)
  - Writing a BigInt library in C
  - Building a simple database
  - Implementing a simple TCP handshake
  - Or any problem statement of your own choosing

## Goals

1. Practice critical thinking in selecting a problem statement, crafting a narrative around the project, and carrying out the actual implementation.
2. As a side benefit, end up with a project you can showcase in interviews. Because of this, it's recommended that each student pick their own individual problem statement.

## Deliverables

Check the following into your GitHub repository:

- **Project proposal**: covering project description, goals, specifications, and design. This must be part of the `README.md` file in the repository, written in Markdown.
- **Project code**: must be written in C. This is non-negotiable.

## Best Practices for the GitHub Repository

- **Use a `.gitignore` file**: Never commit compiled binaries (`.o`, `.obj`), executable files, or build artifacts. Ignore IDE-specific folders (`.vscode/`, `.vs/`) and OS files (`.DS_Store`).
- **Organize folders cleanly**: Separate source files and headers — `.c` files in a `src/` directory, `.h` files in an `include/` directory — keeping the root folder tidy.
- **Provide a Makefile or CMakeLists.txt**: Include a build automation file so anyone can compile the project by running `make` or building via CMake.
- **README.md**: Explain what the C project does, list prerequisites, and provide step-by-step compile/run instructions. Include a quick usage or command-line example if applicable.
- **Add a License**: Choose an open-source license (e.g. MIT) so others know how they can use or modify the code.
- **Write meaningful commit messages**: Describe what changed and why (e.g. "Fix memory leak in parser" instead of "Update file.c").
- **Commit small and often**: Break work into logical, atomic commits rather than pushing the entire working program in one massive update.
- **Avoid committing sensitive data**: Double-check that no hardcoded passwords, API keys, or private configurations are present in the code.
