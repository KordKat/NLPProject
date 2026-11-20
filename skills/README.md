# STEM OS skills

Each `*.md` file in this folder is a built-in **skill**: task-specific instructions that KeawGPT
retrieves only when a request needs them (see `backend/app/services/skills.py`).

```markdown
---
name: my-skill                 # lowercase, digits, hyphens
title: Human-readable title
description: One or two sentences — what the skill teaches and when it applies (used for retrieval).
triggers: keyword one, keyword two   # words/phrases that strongly indicate the skill
tools: run_python, create_file      # existing STEM OS tools the skill relies on
---

# Instructions for the agent …
```

Users can add their own skills (private to them) and hide built-in ones in **Settings → Skills**.
This README is not loaded as a skill.
