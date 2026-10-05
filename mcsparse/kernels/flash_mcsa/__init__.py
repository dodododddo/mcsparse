"""Token-level sparse attention kernels for MiniMax-H3.

The H3 dispatcher constructs TokenSparseConfig and TokenSparseContext directly.
Only token selection and the CuTeDSL gather-KV path are exposed here.
Submodules are deliberately not imported at package load time, so environment
inspection and pure-logic tests also work without PyTorch installed.
"""

__version__ = "0.1.0"

__all__: list[str] = []
