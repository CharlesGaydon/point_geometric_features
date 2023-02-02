import os
import sys
from typing import Any, Optional

filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filepath)

import src.libpgeof as pgeof
import numpy as np
import faiss
import laspy
import argparse

# Default parameters
GLOBAL_DEFAUT_K = 16  # k nearest neighboors to use
K_MIN_GEOF = 1  # min k for geometric features - but will equal to 16 here.
VERBOSE_GEOF = True

EXTRA_DIMS_NAMES = [
    "linearity",
    "planarity",
    "scattering",
    "verticality",
    "normal_x",
    "normal_y",
    "normal_z",
    "length",
    "surface",
    "volume",
    "curvature",
]
EBP = laspy.point.format.ExtraBytesParams
LAS_DATA_EXTRA_DIMS = [EBP(name=dim_name, type="float") for dim_name in EXTRA_DIMS_NAMES]


def get_geof_from_xyz(xyz, k: int = GLOBAL_DEFAUT_K, gpu: Optional[int] = None):
    """Features have shape [N, 11]:
     0 - linearity
     1 - planarity
     2 - scattering
     3 - verticality
     4 - normal_x
     5 - normal_y
     6 - normal_z
     7 - length
     8 - surface
     9 - volume
    10 - curvature
    """

    num_points = len(xyz)
    xyz = xyz.astype("float32")
    xyz = np.ascontiguousarray(xyz)

    index = faiss.IndexFlatL2(3)
    if gpu is not None:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, gpu, index)
        print("Using GPU instead of CPU.")

    index.add(xyz)
    _, nn = index.search(xyz, k)
    del index

    nn = nn.flatten().astype("uint32")
    nn = np.ascontiguousarray(nn)

    nn_ptr = np.r_[0, [k] * num_points].cumsum().astype("uint32")
    nn_ptr = np.ascontiguousarray(nn_ptr)

    geof = pgeof.compute_geometric_features(xyz, nn, nn_ptr, K_MIN_GEOF, VERBOSE_GEOF)
    return geof


def main(in_f, out_f, k: int = GLOBAL_DEFAUT_K, gpu: Optional[int] = None):
    las = laspy.read(in_f)
    # Need to use internal representation (offset+scale) to avoid rounding errors when casting to float32.
    geof = get_geof_from_xyz(np.stack([las.X, las.Y, las.Z]).transpose(), k=k, gpu=gpu)
    las.add_extra_dims(LAS_DATA_EXTRA_DIMS)
    for idx, dim_name in enumerate(EXTRA_DIMS_NAMES):
        las[dim_name] = geof[:, idx]
    las.write(out_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Point Geometric Features - main",
        description="Computes 11 geometric features from and to a LAS point cloud.",
        epilog="Success.",
    )
    parser.add_argument(
        "--in_f",
        default="tests/data/small_single_house.las",
        help="Input LAS File. Extension can be las or laz.",
    )
    parser.add_argument(
        "--out_f",
        default="tests/data/small_single_house.example_output.las",
        help="Output LAS File. Extension can be las or laz.",
    )
    parser.add_argument(
        "--k",
        default=GLOBAL_DEFAUT_K,
        help="Number of neighboors to use when computing geometric features (at least 16 is advised)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Set to a a GPU number (i.e. 0 for the first GPU) to use GPU when searching for k-nn.",
    )
    args = parser.parse_args()

    main(args.in_f, args.out_f, k=args.k, gpu=args.gpu)
