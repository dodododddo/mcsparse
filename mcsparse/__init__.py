"""Token-sparse attention for MiniMax-H3.

    from mcsparse.h3.context_parallel import with_ulysses_sparse
    h3.dispatch_attention_fn = with_ulysses_sparse(
        h3.dispatch_attention_fn, kind="mcsa_token", topk=0.15)

or from the command line:

    mcsparse run --prompt-file examples/prompts.jsonl --index 0
"""
__version__ = "0.1.0"
