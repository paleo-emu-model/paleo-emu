---
title: 'PaleoEmu: a fast Gaussian Process mean state climate model emulator in Python'
tags:
  - Python
  - climate dynamics
  - paleoclimate
authors:
  - name: Xin Ren
    orcid: 0000-0000-0000-0000
    corresponding: true  
    equal-contrib: true #?
    affiliation: 1
  - name: Joost de Vries
    orcid: 0000-0000-0000-0000
    equal-contrib: true  #?
    affiliation: 1
  - name: Charles Williams
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Fanny M. Monteiro
    orcid: 0000-0000-0000-0000
    affiliation: "1, 2"    
  - names: Daniel J. Lunt
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: BRIDGE
   index: 1
   ror: 000000000
 - name: Tromso
   index: 2
   ror: 000000000


date: 13 August 2017
bibliography: paper.bib
---

# Summary
[comment]: <> (A description of the high-level functionality and purpose of the software for a diverse, non-specialist audience.)

(Paleo)climate models are a key tool for understanding our climate, allowing researchers to explore interaction between Earth's systems (atmosphere, oceans, land, ice) and key climate drivers such as greenhouse gasses. Climate models use large sets of mathematical equations built on fundamental laws of physics to represent key earth system processes, and provided detailed scenarios for temperature, wind, ice cover and other properties for past, present and future climates. While climate models vary greatly in complexity, even relatively simple and low resolution models require significant compute which limit's the number of scenarios (e.g. time periods) for which a given model is applied and can also limit inferences of underlying uncertainties.

Emulators (or `surrogate models`) which approximate climate model behavior at a fraction of the computational cost have emerged as a key method to resolve these challenges. While climate model emulators are varied in scope and mechanisms, Gaussian Process regressors in particular have emerged has a highly popular and statistically robust approach. While several research works have implemented such emulators in various contexts and programming languages, to-date, user-facing implementations designed with reproducibility in mind are notably missing. 

`PaleoEmu` thus aims to provide an easy-to-use and highly reproducible emulation framework for climate models. The software achieves this by utilizing Python, and relying on it's associated and well supported machine learning framework `scikit-learn`. All features in the package are thoroughly tested using continuous integration, and a purpose-build model configuration interface ensures a unified and clear surface for user provided model adjustments. The package includes example-driven documentation, providing emulator examples for the HADCM3 climate model.

Because of it's ease of use and reproducibility, `PaleoEmu` is appropriate to be used by researchers, postgraduate, and undergraduate students alike.


# Statement of need
[comment]: <> (A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work.)

# State of the field
[comment]: <> (A description of how this software compares to other commonly-used packages in the research area. If related tools exist, provide a clear “build vs. contribute” justification explaining your unique scholarly contribution and why existing alternatives are insufficient.)

# Software design
[comment]: <> (An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description.)

# Research impact statement
[comment]: <> (Evidence of realized impact (publications, external use, integrations) or credible near-term significance (benchmarks, reproducible materials, community-readiness signals). The evidence should be compelling and specific, not aspirational.)

# AI usage disclosure
[comment]: <> (Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified.)


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Similar work: 

Work by Tran et al., 2016 [@tran:2016]


# Acknowledgements

We acknowledge contributions from 

# References