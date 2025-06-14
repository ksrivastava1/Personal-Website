---
title: "A Perturbation Bound on the Subspace Estimator from Canonical Projections"
authors:
- admin
- Daniel Pimentel-Alarcón
date: "2022-06-26T00:00:00Z"
doi: "10.1109/ISIT50566.2022.9834816"

# Schedule page publish date (NOT publication's date).
publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "IEEE International Symposium on Information Theory, 2022"
publication_short: "IEEE ISIT 2022"

abstract: This paper derives a perturbation bound on the optimal subspace estimator obtained from a subset of its canonical projections contaminated by noise. This fundamental result has important implications in matrix completion, subspace clustering, and related problems.

# Summary. An optional shortened abstract.
summary: Given only the noisy projections of a subspace U onto lower dimensions, we give a method of reconstructing U based on previous work and an upper bound on the error of estimation alongside experiments. 

tags:
- 
featured: false

# links:
# - name: ""
#   url: ""
url_pdf: http://arxiv.org/pdf/2206.14278v1
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

It is often useful to model data using linear subspaces for computational efficiency, model interpretability, and extensive theoretical framework around estimation error and sensitivity to noisy data. A common problem, however, is that while we build linear models using observed data as proxies for basis vectors (for example, in [Low Rank Matrix Completion]{https://en.wikipedia.org/wiki/Matrix_completion} and [Robust PCA]{https://en.wikipedia.org/wiki/Robust_principal_component_analysis}), data is incomplete. Therefore, we only observe partial components of basis vectors. In this setting, there has been theoretical work done in discovering necessary and sufficient conditions for when reconstruction of a subspace from partially observed basis vectors is possible ([1]{https://arxiv.org/pdf/1410.0633}, [2]{https://arxiv.org/pdf/1407.0900}, [3]{https://arxiv.org/pdf/1808.00616}), this was done in the noiseless-data setting. In this work, we present a novel method of reconstructing the optimal subspace estimator from noisy observations and present an upper bound for our estimation error. We also present experimental results to demonstrate how our method performs under synthetic settings. 

You can find relevant code [here](https://github.com/ksrivastava1/identifying-subspaces). You can also find my ISIT 22 presentation slides on this paper {{% staticref "files/isit_final_slides.pdf" %}} here {{% /staticref %}}.
