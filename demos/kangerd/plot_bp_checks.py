import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
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
        try:
            thickness = chk.load_function(mesh, name="thickness", idx=idx)
        except (KeyError, RuntimeError):
            thickness = None
        try:
            surface = chk.load_function(mesh, name="surface", idx=idx)
        except (KeyError, RuntimeError):
            surface = None
    if velocity.function_space().mesh().topological_dimension == 3:
        da_vel = icepack.depth_average(velocity)
        b_vel = extract_bed(velocity)
        t_vel = extract_surface(velocity)
        if thickness is not None:
            thickness = icepack.depth_average(thickness)
        if surface is not None:
            surface = icepack.depth_average(surface)
        return da_vel, thickness, surface, idx, num_indices, b_vel, t_vel
    else:
        return velocity, thickness, surface, idx, num_indices, velocity, velocity


def checkpoint_metadata(filename):
    with firedrake.CheckpointFile(str(filename), "r") as chk:
        timesteps = np.array(chk.h5pyfile["timesteps"][:])
        num_indices = count_velocity_indices(chk)

    return timesteps, num_indices


def usable_indices(timesteps, num_indices):
    return min(len(timesteps), num_indices)


def matching_time_index(timesteps, time, filename):
    if len(timesteps) == 0:
        raise RuntimeError(f"No timesteps found in {filename}")

    return int(np.argmin(np.abs(timesteps - time)))


def input_tag(filename):
    tag = "-".join(Path(filename).stem.split("-")[2:])
    return tag or "2d"


def model_label(filename):
    tag = Path(filename).stem.split("-")[-1]
    if tag == "3d":
        return "3D SSA"
    if tag[:2] == "bp":
        if len(tag) >= 4:
            return f"3D BP (vdegree={tag.split('v')[-1]})"
        return "3D BP"
    return "2D"


def velocity_magnitude(velocity):
    return firedrake.Function(velocity.sub(0).function_space()).interpolate(
        firedrake.sqrt(firedrake.inner(velocity, velocity))
    )


def symmetric_limits(functions):
    vmax = max(float(np.max(np.abs(function.dat.data_ro))) for function in functions)
    if vmax == 0:
        vmax = 1.0
    return -vmax, vmax


def load_comparison(input_2d, input_3d, idx):
    timesteps_2d, num_2d = checkpoint_metadata(input_2d)
    timesteps_3d, num_3d = checkpoint_metadata(input_3d)
    num_usable_2d = usable_indices(timesteps_2d, num_2d)
    num_usable_3d = usable_indices(timesteps_3d, num_3d)
    if num_usable_2d == 0:
        raise RuntimeError(f"No checkpointed indices found in {input_2d}")
    if num_usable_3d == 0:
        raise RuntimeError(f"No checkpointed indices found in {input_3d}")

    idx_2d_requested = min(idx, num_usable_2d - 1)
    time = timesteps_2d[idx_2d_requested]
    idx_3d_requested = matching_time_index(
        timesteps_3d[:num_usable_3d], time, input_3d
    )
    u_3d_da, h_3d, s_3d, idx_3d, num_3d, u_3d_b, u_3d_t = load(
        input_3d, idx_3d_requested, extruded=True
    )
    try:
        u_2d_da, h_2d, s_2d, idx_2d, num_2d, u_2d_b, u_2d_t = load(
            input_2d, idx_2d_requested
        )
    except RuntimeError:
        u_2d_da, h_2d, s_2d, idx_2d, num_2d, u_2d_b, u_2d_t = load(
            input_2d, idx_2d_requested, extruded=True
        )

    return {
        "u_2d_t": u_2d_t,
        "h_2d": h_2d,
        "s_2d": s_2d,
        "idx_2d": idx_2d,
        "num_2d": num_2d,
        "u_3d_b": u_3d_b,
        "u_3d_t": u_3d_t,
        "h_3d": h_3d,
        "s_3d": s_3d,
        "idx_3d": idx_3d,
        "num_3d": num_3d,
        "time": time,
        "time_3d": timesteps_3d[idx_3d],
    }


def add_map_axes_labels(axes):
    axes.set_aspect("equal")
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")


def movie_fields(comparison):
    velocity = velocity_magnitude(comparison["u_3d_t"])
    u_2d_on_3d = firedrake.Function(comparison["u_3d_t"]).interpolate(
        comparison["u_2d_t"]
    )
    velocity_difference = firedrake.Function(velocity).interpolate(
        velocity - velocity_magnitude(u_2d_on_3d)
    )
    thickness = comparison["h_3d"]
    thickness_difference = firedrake.Function(comparison["h_3d"]).interpolate(
        comparison["h_3d"]
        - firedrake.Function(comparison["h_3d"]).interpolate(comparison["h_2d"])
    )

    return velocity, velocity_difference, thickness, thickness_difference


def movie_limits(input_2d, input_3d, num_frames):
    velocity_vmax = 0.0
    velocity_difference_vmax = 0.0
    thickness_vmax = 0.0
    thickness_difference_vmax = 0.0

    for idx in range(num_frames):
        comparison = load_comparison(input_2d, input_3d, idx)
        velocity, velocity_difference, thickness, thickness_difference = movie_fields(
            comparison
        )
        velocity_vmax = max(velocity_vmax, float(np.max(velocity.dat.data_ro)))
        velocity_difference_vmax = max(
            velocity_difference_vmax,
            float(np.max(np.abs(velocity_difference.dat.data_ro))),
        )
        thickness_vmax = max(thickness_vmax, float(np.max(thickness.dat.data_ro)))
        thickness_difference_vmax = max(
            thickness_difference_vmax,
            float(np.max(np.abs(thickness_difference.dat.data_ro))),
        )

    if velocity_vmax == 0:
        velocity_vmax = 1.0
    if velocity_vmax > 5000:
        velocity_vmax = 5000
    if velocity_difference_vmax == 0:
        velocity_difference_vmax = 1.0
    if velocity_difference_vmax > 250:
        velocity_difference_vmax = 250
    if thickness_vmax == 0:
        thickness_vmax = 1.0
    if thickness_vmax > 1750:
        thickness_vmax = 1750
    if thickness_difference_vmax == 0:
        thickness_difference_vmax = 1.0
    if thickness_difference_vmax > 50:
        thickness_difference_vmax = 50

    return {
        "velocity": (0, velocity_vmax),
        "velocity_difference": (-velocity_difference_vmax, velocity_difference_vmax),
        "thickness": (0, thickness_vmax),
        "thickness_difference": (-thickness_difference_vmax, thickness_difference_vmax),
    }


def plot_movie_frame(fig, input_2d, input_3d, timesteps, idx, limits):
    label_2d = model_label(input_2d)
    label_3d = model_label(input_3d)
    comparison = load_comparison(input_2d, input_3d, idx)
    velocity, velocity_difference, thickness, thickness_difference = movie_fields(
        comparison
    )

    fig.clear()
    axes = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.suptitle(f"Time = {comparison['time']:.6g} yr")
    axes[0, 0].set_title(f"{label_3d} velocity magnitude")
    axes[0, 1].set_title(f"{label_3d} - {label_2d}")
    axes[1, 0].set_title(f"{label_3d} thickness")
    axes[1, 1].set_title(f"{label_3d} - {label_2d}")

    add_map_axes_labels(axes[0, 0])
    color = tripcolor(
        velocity,
        axes=axes[0, 0],
        cmap="Reds",
        vmin=limits["velocity"][0],
        vmax=limits["velocity"][1],
        num_sample_points=4,
    )
    fig.colorbar(color, ax=axes[0, 0], label="Velocity magnitude (m / yr)", extend="max")

    add_map_axes_labels(axes[0, 1])
    diff_color = tripcolor(
        velocity_difference,
        axes=axes[0, 1],
        cmap="PuOr",
        vmin=limits["velocity_difference"][0],
        vmax=limits["velocity_difference"][1],
        num_sample_points=4,
    )
    fig.colorbar(
        diff_color,
        ax=axes[0, 1],
        label="Velocity magnitude difference (m / yr)",
        extend="both",
    )

    add_map_axes_labels(axes[1, 0])
    color = tripcolor(
        thickness,
        axes=axes[1, 0],
        cmap="viridis",
        vmin=limits["thickness"][0],
        vmax=limits["thickness"][1],
        num_sample_points=4,
    )
    fig.colorbar(color, ax=axes[1, 0], label="Thickness (m)", extend="max")

    add_map_axes_labels(axes[1, 1])
    diff_color = tripcolor(
        thickness_difference,
        axes=axes[1, 1],
        cmap="PiYG",
        vmin=limits["thickness_difference"][0],
        vmax=limits["thickness_difference"][1],
        num_sample_points=4,
    )
    fig.colorbar(diff_color, ax=axes[1, 1], label="Thickness difference (m)", extend="both")


def plot_bp_movie(input_2d, input_3d, timesteps, num_frames, output, fps):
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("Matplotlib cannot find ffmpeg, which is required for MP4 output")

    limits = movie_limits(input_2d, input_3d, num_frames)
    fig = plt.figure(figsize=(10.0, 8.0), constrained_layout=True)
    writer = animation.FFMpegWriter(fps=fps, metadata={"artist": "plot_bp_checks.py"})

    with writer.saving(fig, output, dpi=150):
        for idx in range(num_frames):
            plot_movie_frame(fig, input_2d, input_3d, timesteps, idx, limits)
            fig.canvas.draw()
            writer.grab_frame()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--input-2d", default="kangerdlugssuaq-year1-3d.h5")
    parser.add_argument("--input-3d", default="kangerdlugssuaq-year1-bp.h5")
    parser.add_argument("--output", default="compare")
    parser.add_argument("--movie-fps", type=int, default=12)
    parser.add_argument("--movie", action="store_true")
    args, petsc_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *petsc_args]
    global firedrake, icepack, tripcolor, extract_surface, extract_bed
    import firedrake
    import icepack
    from icepackaccs import extract_surface, extract_bed

    try:
        from firedrake.pyplot import tripcolor
    except ModuleNotFoundError:
        from firedrake import tripcolor

    input_2d = Path(args.input_2d)
    input_3d = Path(args.input_3d)

    comparison = load_comparison(input_2d, input_3d, args.idx)
    u_2d_t = comparison["u_2d_t"]
    h_2d = comparison["h_2d"]
    s_2d = comparison["s_2d"]
    idx_2d = comparison["idx_2d"]
    num_2d = comparison["num_2d"]
    u_3d_b = comparison["u_3d_b"]
    u_3d_t = comparison["u_3d_t"]
    h_3d = comparison["h_3d"]
    s_3d = comparison["s_3d"]
    idx_3d = comparison["idx_3d"]
    num_3d = comparison["num_3d"]
    time = comparison["time"]
    time_3d = comparison["time_3d"]
    output_prefix = f"{args.output}_{input_tag(args.input_2d)}_{input_tag(args.input_3d)}"
    output = f"{output_prefix}_{idx_2d}"

    if idx_2d != args.idx:
        print(f"{input_2d}: requested idx={args.idx}, using idx={idx_2d} of {num_2d}")
    print(
        f"matched {input_2d} idx={idx_2d} time={time:.6g} yr",
        f"to {input_3d} idx={idx_3d} time={time_3d:.6g} yr",
    )

    velocities = [u_2d_t, u_3d_t]
    thicknesses = [h_2d, h_3d]
    surfaces = [s_2d, s_3d]
    velocity_difference = firedrake.Function(u_2d_t).interpolate(u_2d_t - firedrake.Function(u_2d_t).interpolate(u_3d_t))
    has_thickness = all(thickness is not None for thickness in thicknesses)
    has_surface = all(surface is not None for surface in surfaces)
    if has_thickness:
        thickness_difference = firedrake.Function(h_2d).interpolate(h_2d - firedrake.Function(h_2d).interpolate(h_3d))
    if has_surface:
        surface_difference = firedrake.Function(s_2d).interpolate(s_2d - firedrake.Function(s_2d).interpolate(s_3d))
    label_2d = model_label(args.input_2d)
    label_3d = model_label(args.input_3d)

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

    sliding_velocities = [u_3d_t, u_3d_b]
    sliding_difference = firedrake.Function(u_3d_t).interpolate(u_3d_t - u_3d_b)
    sliding_labels = [
        f"{label_3d} top, idx={idx_3d}",
        f"{label_3d} bed, idx={idx_3d}",
        f"{label_3d} top - bed",
    ]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(12.0, 6.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    vmin, vmax = symmetric_limits(sliding_velocities)
    dvmin, dvmax = symmetric_limits([sliding_difference])
    for col, label in enumerate(sliding_labels):
        axes[0, col].set_title(label)

    for row, (component_label, component) in enumerate(components):
        colors = []
        for col, velocity in enumerate(sliding_velocities):
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
            sliding_difference.sub(component),
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

    fig.savefig(output + f"-sliding.pdf", bbox_inches="tight")

    outputs = [
        f"{output}-vel.pdf",
        f"{output}-sliding.pdf",
    ]
    if has_thickness:
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
        outputs.append(f"{output}-thick.pdf")
    else:
        print("skipping thickness plot because thickness is missing from an input")

    if has_surface:
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(12.0, 4),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        vmax = max(float(np.max(np.abs(surface.dat.data_ro))) for surface in surfaces)
        vmin = 0
        dhmin, dhmax = symmetric_limits([surface_difference])
        for col, label in enumerate(labels):
            axes[col].set_title(label)

        colors = []
        for col, surface in enumerate(surfaces):
            axes[col].set_aspect("equal")
            axes[col].set_xlabel("x (m)")
            axes[col].set_ylabel("y (m)")
            color = tripcolor(
                surface,
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
            surface_difference,
            axes=axes[2],
            cmap="PiYG",
            vmin=dhmin,
            vmax=dhmax,
            num_sample_points=4,
        )

        fig.colorbar(
            colors[0],
            ax=axes[:2],
            label="Surface (m)",
            shrink=0.92,
        )
        fig.colorbar(
            diff_color,
            ax=axes[2],
            label="Surface difference (m)",
            shrink=0.92,
        )

        fig.savefig(output + f"-surf.pdf", bbox_inches="tight")
        outputs.append(f"{output}-surf.pdf")
    else:
        print("skipping surface plot because surface is missing from an input")
    if args.movie:
        timesteps_2d, movie_num_2d = checkpoint_metadata(input_2d)
        num_movie_frames = usable_indices(timesteps_2d, movie_num_2d)
        movie_output = f"{output_prefix}.mp4"
        plot_bp_movie(
            input_2d,
            input_3d,
            timesteps_2d[:num_movie_frames],
            num_movie_frames,
            movie_output,
            args.movie_fps,
        )
        outputs.append(movie_output)
    print(
        "wrote",
        *outputs,
    )

if __name__ == "__main__":
    main()
