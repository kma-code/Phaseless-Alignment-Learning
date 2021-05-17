# Supplementary experiments to Pseudo-Backpropagation and RtDeel in Jupyter

This repo contains some experiments and explanations of the pseudo-backpropagation algorithm.

Pseudo-backprop implements dendritic error learning in discrete time steps.
The algorithm is implemented in https://github.com/unibe-cns/pseudoBackprop

This repo:
- B circuit.ipynb: example implementation of the learning rule for backwards weights B.
Here, the forward weights W are fixed and B is learnt by minimizing the mismatch energy.
The notebook shows how B converges to the data-specific pseudoinverse of W.