# CallumJMac.github.io

Personal website and blog, built with [AstroPaper](https://github.com/satnaing/astro-paper) and deployed to GitHub Pages.

## Local development

```bash
npm install
npm run dev -- --host 127.0.0.1
```

## Writing

Blog posts live in `src/data/blog/*.md`. Each post needs YAML frontmatter:

```yaml
---
pubDatetime: 2026-04-25T00:00:00Z
title: "Post title"
slug: post-slug
description: "Short summary."
tags:
  - AI
---
```

## Build

```bash
npm run build
npm run preview
```

## Deploy

Pushes to `master` trigger the GitHub Actions workflow in `.github/workflows/deploy.yml`, which builds the site and deploys to GitHub Pages.

## Comments

Blog comments use [giscus](https://giscus.app/) backed by GitHub Discussions. Already configured in `src/layouts/PostDetails.astro`.

## Analytics

Page views are tracked with [Cloudflare Web Analytics](https://www.cloudflare.com/web-analytics/) (cookie-free). Dashboard at [dash.cloudflare.com](https://dash.cloudflare.com/).
