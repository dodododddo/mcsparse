"""mcsparse command line.

    mcsparse run    --prompt-file examples/prompts.jsonl --index 0
    mcsparse info                       what this machine can run
    mcsparse examples                   list the bundled prompts

`run` on several GPUs goes through torchrun:

    torchrun --nproc_per_node=7 -m mcsparse.pipeline --prompt-file ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_examples() -> str:
    """Locate prompts.jsonl, installed or from a source checkout.

    The packaged copy lives inside the package (mcsparse/examples/) because
    package-data cannot reference a path outside it. The repo-root copy is
    what an editable install and the README's command line point at, so both
    are checked and the packaged one wins.
    """
    for path in (os.path.join(_HERE, "examples", "prompts.jsonl"),
                 os.path.join(os.path.dirname(_HERE), "examples",
                              "prompts.jsonl")):
        if os.path.exists(path):
            return path
    return os.path.join(_HERE, "examples", "prompts.jsonl")


EXAMPLES = _find_examples()


def cmd_info(_args) -> int:
    print("mcsparse\n")
    # A missing torch used to return here, hiding every other line -- which is
    # the opposite of what you want from `info` on a machine you are still
    # setting up. Report it and carry on through the rest of the checks.
    try:
        import torch
    except ImportError:
        torch = None
        print("  torch                  MISSING  install it first")

    if torch is not None:
        print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            print(f"  {n} GPU(s), sm{cap[0]}{cap[1]}  {name}")
            if cap[0] < 9:
                print("    the CuteDSL kernel targets Hopper (sm90) and newer")
            # What the default layout would pick here.
            if n == 1:
                print("    default layout: single, encoder shared and offloaded")
            else:
                print(f"    default layout: cp over {n - 1} ranks, "
                      f"encoder resident on cuda:{n - 1}")
                print(f"      torchrun --nproc_per_node={n - 1} "
                      f"-m mcsparse.pipeline ...")

    # flash_attn_interface is REQUIRED for token-sparse attention, not optional:
    # the calibrate step needs full attention plus its LSE in one pass, and FA3
    # is the only implementation here that returns both. --no-fa3 still works
    # for --attention dense.
    missing = []
    for mod, why in (("cutlass", "CuteDSL, the sparse kernel"),
                     ("cuda.bindings.driver", "CUDA driver API, used by CuteDSL"),
                     ("triton", "block probabilities"),
                     ("quack", "CuteDSL helpers (pip: quack-kernels)"),
                     ("diffusers", "the H3 pipeline"),
                     ("flash_attn_interface", "FA3 -- required for "
                                              "token_sparse (calibrate needs "
                                              "output+LSE in one pass)")):
        try:
            __import__(mod)
            print(f"  {mod:22} ok    {why}")
        except ImportError:
            print(f"  {mod:22} MISSING  {why}")
            missing.append(mod)
    if "flash_attn_interface" in missing:
        print("\n  without FA3: --attention dense --no-fa3 works; "
              "--attention token_sparse does not")
    return 1 if (torch is None or missing) else 0


def cmd_examples(_args) -> int:
    if not os.path.exists(EXAMPLES):
        print(f"no examples at {EXAMPLES}", file=sys.stderr)
        return 1
    print(f"{EXAMPLES}\n")
    for line in open(EXAMPLES, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        print(f"  [{r['index']}] {r.get('name', '?'):16} "
              f"{len(r['ir']):>5} chars   {r.get('note', '')}")
    print(f"\n  mcsparse run --prompt-file {EXAMPLES} --index 0")
    return 0


def cmd_run(argv) -> int:
    from mcsparse.pipeline import main as pipeline_main
    # Default to the bundled examples so `mcsparse run` alone does something.
    if not any(a.startswith("--prompt") for a in argv):
        argv = ["--prompt-file", EXAMPLES, "--index", "0"] + list(argv)
    return pipeline_main(argv)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="mcsparse", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="info",
                    choices=("run", "info", "examples"))
    known, rest = ap.parse_known_args(argv[:1] if argv else [])
    if known.command == "run":
        return cmd_run(argv[1:])
    if known.command == "examples":
        return cmd_examples(None)
    return cmd_info(None)


if __name__ == "__main__":
    raise SystemExit(main())
