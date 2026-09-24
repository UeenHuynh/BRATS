# Workspace inventory — 2026-09-07

> **Updated after remediation on 2026-09-07.** The storage figures and broken-link counts below are the initial snapshot. Data-authority decisions and P0--P3 membership have since changed as recorded here and in `docs/`.

## Current resolved status

- Canonical raw view: 500 train + 70 validation, unique case IDs, no missing manifest paths.
- `BraTS-MEN-RT-0402-1` is an override from standalone Synapse patch `syn64826221`; the training-v2 image is retained only as immutable source history.
- P0--P3 now each contain 500 training images, 500 labels, and 70 validation images.
- Job `20932` generated and safely installed corrected 0402 into all profiles.
- Job `20935` smoke-tested legacy and new P1 preprocessing on the same corrected input.
- Legacy outputs remain in place for audit but are affected by duplicate/overwrite, split, and zero-background clipping issues.

## Storage snapshot

- Workspace apparent/on-disk size: approximately 178 GB.
- Filesystem free space at inspection time: approximately 559 GB.
- Largest entries:
  - `data1 `: 106 GB, including both archives and extracted profiles.
  - `nnunet_preprocessed/`: 29 GB.
  - `data/`: 25 GB.
  - `data_processed/`: 17 GB.
  - `nnunet_results/`: 2.4 GB.

## Existing source and processed data

- Raw BraTS-MEN-RT Train-v2 case directories: 500.
- `data_processed/images/`: 570 NIfTI images.
- `data_processed/masks/`: 500 NIfTI labels.
- Existing nnU-Net dataset declares 500 labeled training cases.
- There are 1,501 broken symbolic links:
  - `nnunet_raw/`: 1,000.
  - `nnunet_inference/`: 501.
- The broken links use the former absolute root `/mnt/data/uyen/BRATS`.

## New preprocessing bundle already present

The directory is literally named `data1 `, with a trailing space. It contains
six multipart Post_BRATS_v1 archives (about 53 GB total) plus extracted P0--P3
datasets (about 53 GB total). Downloading the Drive bundle again is unnecessary
unless integrity verification finds a problem.

Before remediation, all four extracted profiles contained:

- 499 training images.
- 499 training labels.
- 70 unlabeled validation images.
- No unexpected case IDs relative to the repository manifest.

The one case that was missing from every extracted profile was:

`BraTS-MEN-RT-0402-1`

Profile sizes:

- `P0_MEDIAN_ZSCORE`: 17 GB.
- `P1_MEDIAN_CLIP_ZSCORE`: 9.6 GB.
- `P2_MEDIAN_N4_ZSCORE`: 18 GB.
- `P3_MEDIAN_N4_CLIP_ZSCORE`: 9.5 GB.

This discrepancy was resolved by preprocessing corrected 0402 through all four
profiles and validating the final 500/500/70 membership. The paragraph above is
retained to describe the initial state, not the current dataset.

The raw Train-v2 image and standalone 0402 patch are both retained. Their label
payloads are identical, but the T1c payloads differ. Subsequent provenance review
established the standalone Synapse patch as authoritative. The canonical view
therefore replaces, rather than supplements, the training-v2 copy. See
`docs/DATA_PROVENANCE.md` for hashes and evidence.

## New code clone

- Location: `code/brats-research/`.
- Branch: `main` tracking `origin/main`.
- Commit: `d2f9221dad84857bc98a3f88e2153ae7e4ae2ffa`.
- Commit subject: `Document current benchmark results and limitations`.
- Python source passes bytecode compilation with Python 3; this is a syntax
  check only, not a dependency, GPU, preprocessing, or training test.

## Actions deliberately not performed during the initial inventory

- No legacy directories were moved or renamed.
- No existing symlinks were rewritten.
- No data was downloaded from Google Drive.
- No medical-image payload was uploaded or transferred.
- No SSH connection or remote job was started.
- No checksums have been generated yet.

## Subsequent actions

- Built `datasets/raw_canonical/` using relative symlinks and rewrote the new-code manifest.
- Generated archive/image checksums and recorded them in `datasets/raw_canonical/PROVENANCE.json`.
- Transferred the corrected 0402 raw case and configuration to the remote workspace.
- Processed 0402 for P0--P3, installed only after success, and validated complete membership.
- Added fail-fast canonical and preprocessed validators plus old--new comparison tooling.
- Did not delete, rename, or overwrite the original archives or legacy results.
