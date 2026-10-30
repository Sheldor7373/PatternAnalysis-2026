# Data-free confidence assembly utilities

These helpers are extracted from the completed local source-and-candidate
assembly workflow. They contain no reports, generated candidates, labels or
machine-specific paths. `source_index.py` builds a local metadata/byte-offset
index, with duplicate-ID, fingerprint, staleness and selected-row checks.
`assembly_core.py` provides the fixed component split, conflicting-pair exclusion,
hash-verified repair and resume checks. `schema.example.json` declares the two-text
input invariant and metadata exclusions.

The exact local shard orchestration is not included. An end-to-end assembler must
bind its configured sources and repair lineage explicitly; these helpers alone
are not a claim that an arbitrary archive is fully validated. Synthetic labels
remain weak and unverified, and report/image components do not establish patient
independence. Indexes, generated records and weights remain outside Git.

From this directory run `python -m unittest test_source_index test_assembly_core`.
Seven tests use invented records. Their passing result establishes structural
behaviour, not semantic correctness of generated examples.
