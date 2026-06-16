# Workflow Guidelines — ManiAlimohammadi

## Git Identity

- **Profile repo** (`ManiAlimohammadi/ManiAlimohammadi`, branch `claude/lucid-darwin-72czek`):
  Use `noreply@anthropic.com` / `Claude` as required by the stop hook.

- **Every other (standalone) repo** (e.g. `digit-recognition`, any future project repo):
  Always set before committing:
  ```bash
  git config user.email mani.nani.13.83@gmail.com
  git config user.name ManiAlimohammadi
  ```
  This ensures the user's profile picture appears on GitHub commits and Contributors.

## GitHub Push

1. Try `git push` normally first.
2. If it fails with 403/permission error, use `mcp__github__push_files` tool.
3. If that also fails, ask the user for a Personal Access Token (scope: `repo` only).
4. After the session, remind the user to revoke the token from GitHub Settings.

## Project Structure

- Never create a new project inside the profile repo folder (`ManiAlimohammadi/ManiAlimohammadi`).
- Always create a **dedicated repo** for each new project on GitHub.
- Ask the user to create the repo manually if `mcp__github__create_repository` returns 403.
