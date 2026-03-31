# GitHub Pages Setup Guide

The documentation site is published automatically via GitHub Actions when changes are pushed to the `docs` branch.

## Enable GitHub Pages (one-time, repo admin)

1. Go to **Settings → Pages** in the repository.
2. Under **Source**, select **GitHub Actions**.
3. Push any change to the `docs` branch to trigger the first deployment.

## Local Preview

```bash
pip install -r docs-requirements.txt
mkdocs serve
```

Open `http://localhost:8000` in your browser. The site reloads automatically on file changes.

## Adding or Editing Pages

1. All documentation source files live in `.docs/`.
2. Edit or create `.md` files there.
3. Update the `nav:` section in `mkdocs.yml` to include any new pages.
4. Push to the `docs` branch — the site deploys automatically.
