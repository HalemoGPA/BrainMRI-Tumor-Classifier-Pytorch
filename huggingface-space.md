# Deploying to Hugging Face Spaces

This app deploys to Hugging Face Spaces with the **Docker** SDK so the exact
same image works locally and on the Space. We use the modern `hf` CLI
(install it with `brew install huggingface-cli`).

## One-time setup

```bash
hf auth login          # paste a write-scoped token from https://huggingface.co/settings/tokens
hf auth whoami         # confirm
```

Create the Space:

```bash
hf repos create <your-user>/brain-mri-tumor-classifier \
  --type space \
  --space-sdk docker \
  --public
```

(Or create it in the web UI at <https://huggingface.co/new-space> — pick
**Docker** SDK.)

## Required `README.md` frontmatter

Hugging Face Spaces parses YAML frontmatter at the top of `README.md`. Paste
this block as the very first lines of the README **on the Space**
(`https://huggingface.co/spaces/<your-user>/brain-mri-tumor-classifier`):

```yaml
---
title: Brain MRI Tumor Classifier
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: PyTorch CNN + Grad-CAM for brain-tumor MRI classification.
---
```

The local repo `readme.md` does **not** include this frontmatter so it stays a
clean GitHub README; copy the block in only when pushing to the Space.

## Push the code

Two equivalent ways:

**Option A — git remote (recommended for ongoing updates):**

```bash
git remote add space https://huggingface.co/spaces/<your-user>/brain-mri-tumor-classifier
git push space main
```

**Option B — `hf upload` (one-shot full-folder upload):**

```bash
hf upload <your-user>/brain-mri-tumor-classifier . . \
  --repo-type space \
  --commit-message "Deploy app"
```

The Space auto-builds the Docker image from [Dockerfile](Dockerfile) and starts
Streamlit on port 8501.

## Live link

Once the build finishes, update the **Live demo** section of [readme.md](readme.md)
with the public URL (it will be `https://huggingface.co/spaces/<your-user>/brain-mri-tumor-classifier`).
