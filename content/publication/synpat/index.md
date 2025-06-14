---
title: "SynPAT: A System for Generating Synthetic Physical Theories with Data"
authors:
- admin
- Jonathan Lenchner
- Joao Goncalves
- Lior Horesh
date: "2025-05-30T00:00:00Z"
doi: "10.48550/arXiv.2505.00878"

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

abstract: Machine-assisted methods for discovering new physical laws of nature, starting from a given background theory and data, have recently emerged, and seem to hold the promise of someday advancing our understanding of the physical world. To address these needs, we have developed SynPAT, a system for generating synthetic physical theories comprising (i) a set of consistent axioms, (ii) a symbolic expression that is a consequence of the axioms and the challenge to be discovered, and (iii) noisy data that approximately match the consequence. We also generate theories that do not correctly predict the consequence. We give a detailed description of the inner workings of SynPAT and its various capabilities. We also report on our benchmarking of several open-source symbolic regression systems using our generated theories and data.

# Summary. An optional shortened abstract.
summary: 

tags:
- 
featured: false

# links:
# - name: ""
#   url: ""
url_pdf: https://www.arxiv.org/pdf/2505.00878
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

There have been many works on symbolic regression (finding a formula of best-fit to a dataset without a predetermined form as in linear regression, logistic regression, etc) in context of discovering new physical laws ([1](https://arxiv.org/abs/1905.11481), [2](https://arxiv.org/abs/2305.01582), [3](https://www.science.org/doi/10.1126/science.1165893), [4](https://joss.theoj.org/papers/10.21105/joss.03994), [5](https://arxiv.org/abs/2006.10782)). While these systems all generate formulae that fit data in various contexts, recent breakthroughs have shown that building into the model a framework for exploiting known background theory (encoded as physical axioms) can greatly improve machine-assisted discovery in the scientific context ([1](https://www.nature.com/articles/s41467-023-37236-y), [2](https://www.nature.com/articles/s41467-024-50074-w)). These new systems demonstrate an important new direction for machine-assisted discovery in science to more directly account for background theory in the search in addition to building in heuristics into a model. With this new direction comes also a new need for benchmark datasets to understand the performance of machine-assisted discovery models moving forward. In this work, we present SynPat, a method for generating dimensionally consistent synthetic physical theories which contain i. a list of axioms (encoded as polynomials and ordinary differential equations), ii. a consequence polynomial / ODE of the axioms that is to be discoverred, iii. numeric datasets for the both the axiom systems and consequence phenomena, iv. alternate incorrect axiom systems to test these methods for situations where we do not have complete or correct theory. 

You can find relevant code and dataset [here](https://github.com/jlenchner/theorizer). You can also find the dataset on [huggingface](https://huggingface.com/datasets/Karan0901/synpat-dataset)
