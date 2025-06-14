---
title: "Generative Modeling for Mathematical Discovery"
authors:
- admin
- Jordan S. Ellenberg
- Cristofero S. Fraser-Taliente
- Thomas R. Harvey
- Andrew V. Sutherland
date: "2025-03-14T00:00:00Z"
doi: "10.48550/arXiv.2503.11061"

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

abstract: We present a new implementation of the LLM-driven genetic algorithm {\it funsearch}, whose aim is to generate examples of interest to mathematicians and which has already had some success in problems in extremal combinatorics. Our implementation is designed to be useful in practice for working mathematicians; it does not require expertise in machine learning or access to high-performance computing resources. Applying {\it funsearch} to a new problem involves modifying a small segment of Python code and selecting a large language model (LLM) from one of many third-party providers. We benchmarked our implementation on three different problems, obtaining metrics that may inform applications of {\it funsearch} to new problems. Our results demonstrate that {\it funsearch} successfully learns in a variety of combinatorial and number-theoretic settings, and in some contexts learns principles that generalize beyond the problem originally trained on.

# Summary. An optional shortened abstract.
summary: 

tags:
- 
featured: false

# links:
# - name: ""
#   url: ""
url_pdf: https://www.arxiv.org/pdf/2503.11061
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

In this paper, we provide an implementation of the LLM-driven genetic algorithm [*funsearch*](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), designed to genetically evolve python functions by sampling from third party LLMs to construct examples of interest to mathematicians. The code is designed to be useful for working mathematicians. It does not require machine learning expertise or high-performance computing resources. We benchmark the implementation on various LLMs for three problems in combinatorics and number theory. We also explore in depth how the model can be used to potentially learn principles that generalize beyond the training distribution through the example of finding large isosceles-free subsets of an integer lattice. 

You can find relevant code [here](https://github.com/kitft/funsearch). 
