# Archer's MetaQA integration draft

This folder is for group review. It is **not** the final submitted notebook.
`Archer_MetaQA_Integration_Draft.ipynb` combines a vanilla MetaQA data analysis
with an evaluation of a locally repaired version of the group's baseline at
`main` commit `0f5a3fb`. The saved figures and metrics come from that local
experiment, not from the unchanged online baseline.

The group should review the code, claims, and authorship before adopting any
part for submission. Confirm each member's contribution in the final notebook.

## Reproduce the experiment

1. Obtain the public MetaQA `kb.txt`, vanilla `qa_{train,dev,test}.txt` for
   1/2/3-hop, and the corresponding `qa_{train,dev,test}_qtype.txt` files.
   Keep the data outside Git and retain the dataset's license and citation.
2. On a clean checkout of group commit `0f5a3fb`, apply
   `notebooks/baseline-vanilla-local-repair.patch` from the repository root.
3. Set `METAQA_ROOT` to the folder containing `kb.txt` and the hop folders.
   Set `COMP9444_REPO_ROOT` to the patched group checkout.
4. Use Python 3.13+, NumPy, scikit-learn, and Matplotlib. Run the notebook
   cells in order and check the outputs against the saved draft.

The dataset and local machine paths are intentionally absent from this branch.
The group should decide whether to adopt the repair and integrate the material
into its **single** final notebook.
