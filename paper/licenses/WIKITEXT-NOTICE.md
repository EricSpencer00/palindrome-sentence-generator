# WikiText attribution

The saved natural-language samples in `data/mirror-cost/results.json` are
derived from the `wikitext-2-raw-v1` configuration of Salesforce's WikiText
dataset. The dataset card identifies the material as English text from
Wikipedia articles and lists both `cc-by-sa-3.0` and `gfdl` licenses.

- Dataset card: https://huggingface.co/datasets/Salesforce/wikitext
- Associated paper: Stephen Merity, Caiming Xiong, James Bradbury, and Richard
  Socher. *Pointer Sentinel Mixture Models*. arXiv:1609.07843 (2016).
- Frozen dataset snapshot: `b08601e04326c79dfdd32d625aee71d232d685c3`
- Source parquet SHA-256:
  `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`

The release contains 900 target-conditioned sampled spans and normalized
letter streams, not the complete WikiText parquet. Preserve this attribution
and the applicable share-alike/license notices when redistributing them.
