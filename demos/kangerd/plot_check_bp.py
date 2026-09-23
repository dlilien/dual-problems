import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import firedrake
from icepackaccs import extract_bed, extract_surface

try:
    from firedrake.pyplot import tripcolor
except ModuleNotFoundError:
    from firedrake import tripcolor


def symmetric_limits(functions):
    vmax = max(float(np.max(np.abs(function.dat.data_ro))) for function in functions)
    if vmax == 0:
        vmax = 1.0
    return -vmax, vmax


def load_velocities(filename, mesh_name):
    with firedrake.CheckpointFile(str(filename), "r") as chk:
        if mesh_name is None:
            mesh_names = chk._get_mesh_name_topology_name_map()
            mesh_name = next(
                name for name, topology in mesh_names.items() if "extruded" in topology
            )
        mesh = chk.load_mesh(mesh_name)
        u = chk.load_function(mesh, name="velocity_dual")
        u_in = chk.load_function(mesh, name="velocity_primal")
        basal_stress = chk.load_function(mesh, name="basal_stress")
        K = chk.load_function(mesh, name="sliding_coefficient")
        q = chk.load_function(mesh, name="log_friction")

    return u, u_in, basal_stress, K, q


def load_primal_velocity(filename):
    try:
        with firedrake.CheckpointFile(str(filename), "r") as chk:
            mesh = chk.load_mesh()
            velocity = chk.load_function(mesh, name="velocity")
            v_b = velocity, v_t = velocity
    except:
        with firedrake.CheckpointFile(str(filename), "r") as chk:
            mesh = chk.load_mesh("kangerdlugssuaq")
            velocity = chk.load_function(mesh, name="velocity")
            v_b = extract_bed(velocity)
            v_t = extract_surface(velocity)
    return v_b, v_t


def add_map_axes_labels(axes):
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")


def plot_velocity_comparison(left, center, left_label, center_label, output):
    difference = firedrake.Function(left).interpolate(center - left)
    labels = [left_label, center_label, f"{center_label} - {left_label}"]
    components = [("x velocity", 0), ("y velocity", 1)]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12.0, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    vmin, vmax = symmetric_limits([left, center])
    dvmin, dvmax = symmetric_limits([difference])
    for col, label in enumerate(labels):
        axes[0, col].set_title(label)

    for row, (component_label, component) in enumerate(components):
        colors = []
        for col, velocity in enumerate([left, center]):
            add_map_axes_labels(axes[row, col])
            color = tripcolor(
                velocity.sub(component),
                axes=axes[row, col],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                num_sample_points=4,
            )
            colors.append(color)

        add_map_axes_labels(axes[row, 2])
        diff_color = tripcolor(
            difference.sub(component),
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

    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_basal_stress(basal_stress, output):
    if basal_stress.function_space().mesh().geometric_dimension == 3:
        basal_stress = extract_bed(basal_stress)
    components = [("x basal stress", 0), ("y basal stress", 1)]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9.0, 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    vmin, vmax = -1, 1

    for col, (component_label, component) in enumerate(components):
        axes[col].set_title(component_label)
        add_map_axes_labels(axes[col])
        color = tripcolor(
            basal_stress.sub(component),
            axes=axes[col],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            num_sample_points=4,
        )

    fig.colorbar(
        color,
        ax=axes,
        label="Basal stress (MPa)",
        shrink=0.92,
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

def plot_sliding_coeffs(K, q, output):
    if K.function_space().mesh().geometric_dimension == 3:
        K = extract_bed(K)
        q = extract_bed(q)
    
    q = firedrake.Function(K.function_space()).interpolate(firedrake.exp(3 * q))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(9.0, 4.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    components = [("K", K, "viridis"), ("q", q, "inferno")]
    for col, (component_label, component, cmap) in enumerate(components):
        axes[col].set_title(component_label)
        add_map_axes_labels(axes[col])
        color = tripcolor(
            component,
            axes=axes[col],
            cmap=cmap,
            vmin=0,
            vmax=1e6,
            num_sample_points=4,
        )
        fig.colorbar(
            color,
            ax=axes[col],
            label=component_label,
            shrink=0.92,
        )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="kangerdlugssuaq-check-bp.h5")
    parser.add_argument("--input-primal", default="kangerdlugssuaq-initial-bp.h5")
    parser.add_argument("--mesh-name", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_filename = Path(args.input)
    output = args.output
    if output is None:
        output = input_filename.with_suffix("").name

    u, u_in, basal_stress, K, q = load_velocities(input_filename, args.mesh_name)
    _, u_primal_t = load_primal_velocity(args.input_primal)
    u_t = extract_surface(u)
    Q = u_t.function_space()
    u_b = firedrake.Function(Q).interpolate(extract_bed(u))
    u_t_in = firedrake.Function(Q).interpolate(extract_surface(u_in))
    u_b_in = firedrake.Function(Q).interpolate(extract_bed(u_in))
    u_t_primal = firedrake.Function(Q).interpolate(u_primal_t)

    plot_velocity_comparison(
        u_b_in,
        u_t,
        r"$u_{in}$ surface",
        r"$u$ surface",
        f"{output}-vel.pdf",
    )
    plot_velocity_comparison(
        u_b,
        u_t,
        r"$u$ bed",
        r"$u$ surface",
        f"{output}-sliding.pdf",
    )
    plot_velocity_comparison(
        u_b_in,
        u_t_in,
        r"$u_{in}$ bed",
        r"$u_{in}$ surface",
        f"{output}-sliding-in.pdf",
    )
    plot_velocity_comparison(
        u_t_primal,
        u_t,
        "2D",
        r"$u$ surface",
        f"{output}-vel-primal-dual.pdf",
    )
    plot_velocity_comparison(
        u_t_primal,
        u_t_in,
        "2D",
        r"$u_{in}$ surface",
        f"{output}-vel-in-primal.pdf",
    )
    plot_basal_stress(basal_stress, f"{output}-basal-stress.pdf")
    plot_sliding_coeffs(K, q, f"{output}-sliding-coeff.pdf")

    print(
        "wrote",
        f"{output}-vel.pdf",
        f"{output}-sliding.pdf",
        f"{output}-sliding-in.pdf",
        f"{output}-vel-primal-dual.pdf",
        f"{output}-vel-in-primal.pdf",
        f"{output}-basal-stress.pdf",
        f"{output}-sliding-coeff.pdf",
    )


if __name__ == "__main__":
    main()
