---
title: "AI Noether - Bridging the Gap Between Scientific Laws Derived by AI Systems and Canonical Knowledge via Abductive Inference"
authors:
- admin
- Sanjeeb Dash
- Ryan Cory-Wright
- Barry Trager
- Lior Horesh
date: "2025-09-26T00:00:00Z"
doi: "10.48550/arXiv.2509.23004"

# Schedule page publish date (NOT publication's date).
publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "Preprint"
publication_short: "Preprint"

abstract: >
  A core goal in modern science is to harness recent advances in AI and computer processing to automate and
  accelerate the scientific method. Symbolic regression can fit interpretable models to data, but these models
  often sit outside established theory. Recent systems (e.g., AI Descartes, AI Hilbert) enforce derivability
  from prior axioms. However, sometimes new data and associated hypotheses derived from data are not consistent
  with existing theory because the existing theory is incomplete or incorrect. Automating abductive inference
  to close this gap remains open. We propose a solution: an algebraic geometry-based system that, given an
  incomplete axiom system and a hypothesis that it cannot explain, automatically generates a minimal set of
  missing axioms that suffices to derive the axiom, as long as axioms and hypotheses are expressible as
  polynomial equations. We formally establish necessary and sufficient conditions for the successful retrieval
  of such axioms. We illustrate the efficacy of our approach by demonstrating its ability to explain Kepler's
  third law and a few other laws, even when key axioms are absent.

# Summary. An optional shortened abstract.
summary: 

tags:
- 
featured: false

# links:
# - name: ""
#   url: ""
url_pdf: https://arxiv.org/pdf/2509.23004
url_code: ''
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""
---

Machine-assisted scientific discovery has been a popular topic in machine learning. Recent breakthroughs have shown that building a framework for exploiting known background theory (encoded as physical axioms) in addition to data-driven symbolic regression models can greatly improve machine-assisted discovery in the scientific context ([1](https://www.nature.com/articles/s41467-023-37236-y), [2](https://www.nature.com/articles/s41467-024-50074-w)). These methods not only generate hypotheses that fit data, but are also derivable from theory as well. This certificate of derivability, however, is only guaranteed when the background theory itself is complete. If there are essential axioms missing, then while these systems are able to recover the correct hypothesis from data, they are not able to generate a certificate of derivability. Therefore, there is a gap of explainability between machine-generated hypotheses and known theory if the known theory is incomplete. In this paper, we attempt to bridge this gap with AI Noether, a computational framework for abductively inferring missing axioms that are required for an incomplete background theory (encoded as polynomials) to explain a hypothesis. 

You can find relevant code [here](https://github.com/IBM/AI-Noether). 
