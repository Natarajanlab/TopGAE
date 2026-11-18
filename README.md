# TopGAE: A topology-informed regulatory influence-weighted graph autoencoder for gene regulatory network reconstruction from single-cell transcriptomics
Shreyansh Priyadarshi and Kedar Natarajan

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-276DC3)](https://www.r-project.org/)

> **Code accompanying manuscript ** 

---

## Overview

Reconstructing directed gene regulatory networks (GRNs) from single-cell RNA-seq (scRNA-seq) data is challenging due to high sparsity, technical dropout, and cellular heterogeneity, while existing graph models often mask individual transcription factor (TF) influences or overfit incomplete priors. This repository provides TopGAE, a topology-informed graph autoencoder that preserves regulatory asymmetry and jointly reconstructs network links and masked expression profiles by scoring TF-gene edges via allocation coefficients and bounded interaction strengths.


---

## Data Availability

Synthetic and scRNA-seq data are publicly available from the sources listed here as well as the manuscript. The notebooks used for pre-processing data and analysis are included in this repository. No raw data files are committed to this repository.

---

## Citation

```bibtex
@article{priyadarshi2026topgae,
  title={TopGAE: A topology-informed regulatory influence-weighted graph autoencoder for gene regulatory network reconstruction from single-cell transcriptomics},
  author={[Priyadarshi S] and [Natarajan KN]},
  journal={[Under review]},
  year={2026}
}
@software{priyadarshi2026topgae_code,
  title={TopGAE: A topology-informed regulatory influence-weighted graph autoencoder for gene regulatory network reconstruction from single-cell transcriptomics},
  author={[Priyadarshi S] and [Natarajan KN]},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.24352290},
  url={https://doi.org}
}
```

---

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.
