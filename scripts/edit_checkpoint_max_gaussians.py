# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Edit the ``max_gaussians`` optimizer cap stored inside an frgs reconstruct checkpoint.

By default the cap is set to the checkpoint's current Gaussian count, which freezes further
growth on ``frgs resume`` (net-positive refinement steps are skipped) while still allowing
pruning-only steps. The source checkpoint is never modified; a new file is written.
"""

import argparse
import pathlib

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite max_gaussians in an frgs reconstruct checkpoint.")
    parser.add_argument("checkpoint", type=pathlib.Path, help="Path to the source reconstruct_ckpt.pt")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output path (default: alongside source as <stem>_frozen.pt)",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=None,
        help="New cap. Default: freeze at the checkpoint's current Gaussian count.",
    )
    args = parser.parse_args()

    src = args.checkpoint
    dst = args.output if args.output is not None else src.with_name(f"{src.stem}_frozen.pt")
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {dst}")

    # Full load to CPU (not mmap) so the modified state dict can be re-saved.
    state_dict = torch.load(src, map_location="cpu", weights_only=False)

    current_count = int(state_dict["model"]["means"].shape[0])
    new_cap = int(args.max_gaussians) if args.max_gaussians is not None else current_count
    old_cap = state_dict["optimizer"]["config"]["max_gaussians"]
    state_dict["optimizer"]["config"]["max_gaussians"] = new_cap

    torch.save(state_dict, dst)
    print(f"current_gaussians={current_count:,}  old_cap={old_cap:,}  new_cap={new_cap:,}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
