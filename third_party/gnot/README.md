# GNOT source audit

Reference: Hao et al., *GNOT: A General Neural Operator Transformer for Operator
Learning*, ICML 2023, [arXiv:2302.14376v3](https://arxiv.org/abs/2302.14376v3).
The paper links [thu-ml/GNOT](https://github.com/thu-ml/GNOT), whose pinned
reviewed commit is `5ee2e6925a43f9a340a6016bad4da2c82a452cbe`.

The actual GitHub metadata and recursive tree were retrieved. The repository's
`license` field is null and its tree contains no explicit license file. Therefore
this directory does **not** redistribute the upstream implementation or assert
a license for it. `source_audit_v1.json` records retrieval identity and SHA-256
hashes of the reviewed paper, source and metadata.

`src/geophy_set3d/models/baseline_gnot_v1.py` is this project's independent
equation-level implementation and ray-integral adaptation. It preserves GNOT's
heterogeneous normalized operator attention and coordinate-gated experts, but
does not claim to be the original authors' source or their original task.
