# Literature Source Inventory

Verified on 2026-07-14. PDFs are stored for research use; repository links below identify author-provided code only. A missing asset means it was not linked by the paper, publisher page, author project, or a repository owned by an author during this audit.

## Documents

| Work | Primary source | Local PDF | SHA-256 |
|---|---|---|---|
| CAER-Net | [ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Lee_Context-Aware_Emotion_Recognition_Networks_ICCV_2019_paper.html) | `pdf/caer_net_iccv2019.pdf` | `6f4d426bd56b9a8afac9e061c00f4f718ab14e42400fc266cfdb204b22e27fb8` |
| GLAMOR-Net | [arXiv 2111.04129](https://arxiv.org/abs/2111.04129) | `pdf/glamor_net_2021.pdf` | `eb9e7a601d1655984c4d7f648175a0fbfd96bee9ed86e2399ddc29f876fd2b2d` |
| CAHFW-Net | [DOI](https://doi.org/10.3390/ijerph20021400) | `pdf/cahfw_net_2023.pdf` | `68cf482ff3575f8a394a1855afcd672a5cb99a80a65fa4d4b25263f7339e03b1` |
| CCIM | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Context_De-Confounded_Emotion_Recognition_CVPR_2023_paper.html) | `pdf/ccim_cvpr2023.pdf` | `78dd1d167d4ef72f8da34d8f592c199af9815f75320f2289efedffe85f2a7cec` |
| CLEF | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Robust_Emotion_Recognition_in_Context_Debiasing_CVPR_2024_paper.html) | `pdf/clef_cvpr2024.pdf` | `eef88ae826cfa4da2aec94a2b7fd9ed890522ee4ab6af1be23ea57b1a4ff7fb3` |
| DSCT | [ACM DOI](https://doi.org/10.1145/3664647.3680623) | `pdf/dsct_acmmm2024.pdf` | `5f1b8442623d9d9237cb1307715d06e7810ec26b396776f195f7b2cb1b12f105` |
| AGCD-Net | [arXiv 2507.09248](https://arxiv.org/abs/2507.09248) | `pdf/agcd_net_2025_preprint.pdf` | `695477272300758060b5e728b5672b1726de79cbfe13102d32b93e4956ee77ce` |
| EMOTIC | [arXiv 2003.13401](https://arxiv.org/abs/2003.13401) | `pdf/emotic_tpami.pdf` | `92e99c0f08e991f996dc4ee5ff7eacc13681b9042d7bc4316089ad9634a61e14` |
| EmotiCon | [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Mittal_EmotiCon_Context-Aware_Multimodal_Emotion_Recognition_Using_Freges_Principle_CVPR_2020_paper.html) | `pdf/emoticon_cvpr2020.pdf` | `d5f38ddd825959eeed91707e6f14ef34e2dc5a65fc355b12a3301b14c74fc6b9` |

The EmotiCon supplement is `supplements/emoticon_cvpr2020_supplement.pdf` (`918d488fb9612a3e3c89cf91af476900fe7cab6c9dcd4dea87c16eca366f9774`). No other official supplement was exposed by the primary pages.

## Code And Assets

| Work | Official source | Frozen commit | License/assets |
|---|---|---|---|
| GLAMOR-Net | `code_sources/glamor-net` | `c12e1b97aa7354df126795f19402303a00166ec3` | MIT; includes `config.py` and trained weights. |
| CCIM | `code_sources/ccim` | `d6a651f91d1c1c91faca862ddeea915df9314919` | MIT; core `CCIM.py` only, no full training config/checkpoint. |
| EMOTIC | `code_sources/emotic` | `69c3a5106aed08121cd12f6a5b359c745136931e` | Code MIT; dataset restricted to non-commercial research/education. |

No official implementation was found for CAER-Net, CAHFW-Net, CLEF, DSCT, AGCD-Net, or EmotiCon. `third_party/CAER` is an explicitly community-maintained CAER-Net reproduction, not author code. AGCD-Net cites that community repository for its CAER-S protocol, but does not release AGCD-Net code or weights.
