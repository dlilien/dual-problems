import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def count_velocity_indices(chk):
    counts = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return

        dataset_name = name.rsplit("/", maxsplit=1)[-1]
        velocity_names = {"velocity", "firedrake_embedded_velocity"}
        if dataset_name in velocity_names and obj.ndim == 2:
            counts.append(obj.shape[0])

    chk.h5pyfile.visititems(visit)
    if not counts:
        raise RuntimeError(f"No checkpointed velocity dataset found in {chk.filename}")

    return max(counts)


def load(filename, requested_idx, extruded=False):
    filename = str(filename)
    with firedrake.CheckpointFile(filename, "r") as chk:
        num_indices = count_velocity_indices(chk)
        idx = requested_idx if requested_idx < num_indices else num_indices - 1

        if extruded:
            mesh_names = chk._get_mesh_name_topology_name_map()
            mesh_name = next(
                name for name, topology in mesh_names.items() if "extruded" in topology
            )
            mesh = chk.load_mesh(mesh_name)
        else:
            mesh = chk.load_mesh()

        velocity = chk.load_function(mesh, name="velocity", idx=idx)
        thickness = chk.load_function(mesh, name="thickness", idx=idx)

    if velocity.function_space().mesh().topological_dimension() == 3:
        velocity = icepack.depth_average(velocity)
        thickness = icepack.depth_average(thickness)

    return velocity, thickness, idx, num_indices


def load_extruded_velocity(filename, requested_idx):
    filename = str(filename)
    with firedrake.CheckpointFile(filename, "r") as chk:
        num_indices = count_velocity_indices(chk)
        idx = requested_idx if requested_idx < num_indices else num_indices - 1
        mesh_names = chk._get_mesh_name_topology_name_map()
        mesh_name = next(
            name for name, topology in mesh_names.items() if "extruded" in topology
        )
        mesh = chk.load_mesh(mesh_name)
        velocity = chk.load_function(mesh, name="velocity", idx=idx)

    return velocity, idx, num_indices


def surface_bottom_velocities(velocity):
    average = icepack.depth_average(velocity)
    surface = firedrake.Function(average.function_space(), name="surface_velocity")
    bottom = firedrake.Function(average.function_space(), name="bottom_velocity")

    top_nodes = np.asarray(velocity.function_space().boundary_nodes("top"), dtype=int)
    bottom_nodes = np.asarray(velocity.function_space().boundary_nodes("bottom"), dtype=int)
    num_horizontal_nodes = average.dat.data_ro.shape[0]

    if len(top_nodes) == num_horizontal_nodes and len(bottom_nodes) == num_horizontal_nodes:
        surface.dat.data[:] = velocity.dat.data_ro[top_nodes]
        bottom.dat.data[:] = velocity.dat.data_ro[bottom_nodes]
    else:
        surface.dat.data[:] = average.dat.data_ro
        bottom.dat.data[:] = average.dat.data_ro

    return surface, bottom


def difference_function(left, right, name):
    if left.dat.data_ro.shape != right.dat.data_ro.shape:
        raise ValueError(
            f"Cannot difference {left.name()} and {right.name()}: "
            f"{left.dat.data_ro.shape} != {right.dat.data_ro.shape}"
        )

    result = firedrake.Function(right.function_space(), name=name)
    result.dat.data[:] = left.dat.data_ro - right.dat.data_ro
    return result


def symmetric_limits(functions):
    vmax = max(float(np.max(np.abs(function.dat.data_ro))) for function in functions)
    if vmax == 0:
        vmax = 1.0
    return -vmax, vmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=100)
    parser.add_argument("--input-2d", default="steady-state-coarse.h5")
    parser.add_argument("--input-3d", default="steady-state-coarse-bp.h5")
    parser.add_argument("--output", default="compare")
    args, petsc_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *petsc_args]

    output = args.output + f"_{"-".join(args.input_2d.split('/')[-1].split('.')[0].split('-')[2:])}_{"-".join(args.input_3d.split('/')[-1].split('.')[0].split('-')[2:])}_{args.idx}"

    global firedrake, icepack, tripcolor
    import firedrake
    import icepack

    try:
        from firedrake.pyplot import tripcolor
    except ModuleNotFoundError:
        from firedrake import tripcolor

    input_2d = Path(args.input_2d)
    input_3d = Path(args.input_3d)

    u_3d, h_3d, idx_3d, num_3d = load(input_3d, args.idx, extruded=True)
    try:
        u_2d, h_2d, idx_2d, num_2d = load(input_2d, idx_3d)
    except RuntimeError:
        u_2d, h_2d, idx_2d, num_2d = load(input_2d, idx_3d, extruded=True)
    if idx_3d != idx_2d:
        u_3d, h_3d, idx_3d, num_3d = load(input_3d, idx_2d, extruded=True)

    if idx_2d != args.idx:
        print(f"{input_2d}: requested idx={args.idx}, using idx={idx_2d} of {num_2d}")
    if idx_3d != args.idx:
        print(f"{input_3d}: requested idx={args.idx}, using idx={idx_3d} of {num_3d}")

    velocities = [u_2d, u_3d]
    thicknesses = [h_2d, h_3d]
    velocity_difference = difference_function(u_3d, u_2d, name="velocity_difference")
    thickness_difference = difference_function(h_3d, h_2d, name="thickness_difference")
    if args.input_2d.split('/')[-1].split('.')[0].split('-')[-1] == "3d":
        label_2d = "3D SSA"
    elif args.input_2d.split('/')[-1].split('.')[0].split('-')[-1] == "bp":
        label_2d = "3D BP"
    else:
        label_2d = "2D"

    if args.input_3d.split('/')[-1].split('.')[0].split('-')[-1] == "3d":
        label_3d = "3D SSA"
    elif args.input_3d.split('/')[-1].split('.')[0].split('-')[-1] == "bp":
        label_3d = "3D BP"
    else:
        label_3d = "2D"

    labels = [
        f"{label_2d}, idx={idx_2d}",
        f"{label_3d}, idx={idx_3d}",
        f"{label_3d} - {label_2d}",
    ]
    components = [("x velocity", 0), ("y velocity", 1)]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12.0, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    vmin, vmax = symmetric_limits(velocities)
    dvmin, dvmax = symmetric_limits([velocity_difference])
    for col, label in enumerate(labels):
        axes[0, col].set_title(label)

    for row, (component_label, component) in enumerate(components):
        colors = []
        for col, velocity in enumerate(velocities):
            axes[row, col].set_aspect("equal")
            axes[row, col].set_xlabel("x (m)")
            axes[row, col].set_ylabel("y (m)")
            color = tripcolor(
                velocity.sub(component),
                axes=axes[row, col],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                num_sample_points=4,
            )
            colors.append(color)

        axes[row, 2].set_aspect("equal")
        axes[row, 2].set_xlabel("x (m)")
        axes[row, 2].set_ylabel("y (m)")
        diff_color = tripcolor(
            velocity_difference.sub(component),
            axes=axes[row, 2],
            cmap="PuOr",
            vmin=dvmin,
            vmax=dvmax,
            num_sample_points=4,
        )

        fig.colorbar(
            colors[0],
            ax=axes[row, :2],
            label=f"{component_label} (m / yr)",
            shrink=0.92,
        )
        fig.colorbar(
            diff_color,
            ax=axes[row, 2],
            label=f"{component_label} difference (m / yr)",
            shrink=0.92,
        )

    fig.savefig(output + f"-vel.pdf", bbox_inches="tight")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12.0, 4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    vmax = max(float(np.max(np.abs(thickness.dat.data_ro))) for thickness in thicknesses)
    vmin = 0
    dhmin, dhmax = symmetric_limits([thickness_difference])
    for col, label in enumerate(labels):
        axes[col].set_title(label)

    colors = []
    for col, thickness in enumerate(thicknesses):
        axes[col].set_aspect("equal")
        axes[col].set_xlabel("x (m)")
        axes[col].set_ylabel("y (m)")
        color = tripcolor(
            thickness,
            axes=axes[col],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            num_sample_points=4,
        )
        colors.append(color)

    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    diff_color = tripcolor(
        thickness_difference,
        axes=axes[2],
        cmap="PiYG",
        vmin=dhmin,
        vmax=dhmax,
        num_sample_points=4,
    )

    fig.colorbar(
        colors[0],
        ax=axes[:2],
        label="Thickness (m)",
        shrink=0.92,
    )
    fig.colorbar(
        diff_color,
        ax=axes[2],
        label="Thickness difference (m)",
        shrink=0.92,
    )

    fig.savefig(output + f"-thick.pdf", bbox_inches="tight")
    print(
        f"wrote {args.output}-vel-{idx_3d}.pdf, "
        f"{args.output}-thick-{idx_3d}.pdf"
    )


if __name__ == "__main__":
    main()
