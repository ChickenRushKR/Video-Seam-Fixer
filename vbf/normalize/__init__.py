"""Boundary chain-normalization (v1).

The accepted post-hoc method for making cuts between generated clips invisible:
align every clip to a common reference (geometry + colour/exposure + sharpness +
lighting), chained across the whole sequence, then drop the duplicate seam frame.
See ``vbf.normalize.chain.normalize_chain``.
"""

from vbf.normalize.chain import ChainResult, normalize_chain

__all__ = ["normalize_chain", "ChainResult"]
