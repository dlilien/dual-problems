import argparse
from petsc4py import PETSc
import numpy as np
import geojson
import rasterio
import xarray
import firedrake
import firedrake.adjoint
from firedrake import exp, ln, sqrt, assemble, Constant, inner, dx, ds_t, ds_b
import icepack
from icepack.calculus import grad
from icepack2.constants import (
    ice_density as ρ_I, gravity as g, weertman_sliding_law as m
)

parser = argparse.ArgumentParser()
parser.add_argument("--outline")
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--regularization", type=float, default=2.5e3)
parser.add_argument("--output", default="kangerdlugssuaq-initial-bp.h5")
parser.add_argument("--input", default="kangerdlugssuaq-initial.h5")
args = parser.parse_args()

with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh2d = chk.load_mesh()
    q2d = chk.load_function(mesh2d, name="log_friction")
    τ_c = chk.h5pyfile.attrs["mean_stress"]
    u_c = chk.h5pyfile.attrs["mean_speed"]
mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)
mesh.name = "kangerdlugssuaq"

Q_dc = firedrake.FunctionSpace(mesh, "CG", args.degree, vfamily="R", vdegree=0)

Q = firedrake.FunctionSpace(mesh, "CG", args.degree, vfamily="R", vdegree=0)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree, dim=2, vfamily="CG", vdegree=2)
V0 = firedrake.VectorFunctionSpace(mesh, "CG", args.degree, dim=2, vfamily="R", vdegree=0)


q_dc = icepack.lift3d(q2d, Q_dc)
q = firedrake.Function(Q).interpolate(q_dc)

# Read the thickness and surface data
bedmachine = xarray.open_dataset("/Volumes/LaCie/Data/greenland_general/bedmachine/BedMachineGreenland-v6.nc")
h_obs = icepack.interpolate(bedmachine["thickness"], Q)
s_obs = icepack.interpolate(bedmachine["surface"], Q)

def smoothe(q_obs, λ):
    q = q_obs.copy(deepcopy=True)
    J = 0.5 * ((q - q_obs)**2 + Constant(λ)**2 * inner(grad(q), grad(q))) * dx
    F = firedrake.derivative(J, q)
    firedrake.solve(F == 0, q)
    return q

λ = 2e3
h = smoothe(h_obs, λ)
s = smoothe(s_obs, λ)

outline_filename = "/Users/dlilien/Projects/ISMIP7/greenland/dual-problems/demos/kangerd/kangerdlugssuaq.geojson"
if args.outline:
    outline_filename = args.outline
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)
# Read all the raw velocity data
coords = np.array(list(geojson.utils.coords(outline)))
delta = 2.5e3
extent = {
    "left": coords[:, 0].min() - delta,
    "right": coords[:, 0].max() + delta,
    "bottom": coords[:, 1].min() - delta,
    "top": coords[:, 1].max() + delta,
}
measures_filenames = [f"/Volumes/LaCie/Data/greenland_general/velocity/multiyear/greenland_vel_mosaic200_2015_2016_{key}_v02.1.tif" for key in ["vx", "vy", "ex", "ey"]]

velocity_data = {}
for key in ["vx", "vy", "ex", "ey"]:
    filename = [f for f in measures_filenames if key in f][0]
    with rasterio.open(filename, "r") as source:
        window = rasterio.windows.from_bounds(
            **extent, transform=source.transform
        ).round_lengths().round_offsets()
        transform = source.window_transform(window)
        velocity_data[key] = source.read(indexes=1, window=window)

PETSc.Sys.Print("done reading observational data")

# Find the points that are inside the domain and create a point cloud
indices = np.array(
    [
        (i, j)
        for i in range(window.width)
        for j in range(window.height)
        if (
            mesh2d.locate_cell(transform * (i, j), tolerance=1e-8) and
            velocity_data["ex"][j, i] >= 0.0
        )
    ]
)
xs = np.array([transform * idx for idx in indices])
xs = np.hstack([xs, np.ones((len(xs), 1))])
point_set = firedrake.VertexOnlyMesh(
    mesh, xs, missing_points_behaviour="error"
)

Δ = firedrake.FunctionSpace(point_set, "DG", 0)
if hasattr(point_set, "input_ordering"):
    Δ_input = firedrake.FunctionSpace(point_set.input_ordering, "DG", 0)
    def gridded_to_point_set(field):
        f_input = firedrake.Function(Δ_input)
        with f_input.dat.vec as v:
            lo, hi = v.getOwnershipRange()
            vals = field[indices[:, 1], indices[:, 0]]
        v.setArray(vals[lo:hi])
        # f_input.dat.data[:] = field[indices[:, 1], indices[:, 0]]
        f_output = firedrake.Function(Δ)
        f_output.interpolate(f_input)
        return f_output
else:
    def gridded_to_point_set(field):
        f = firedrake.Function(Δ)
        with f.dat.vec as v:
            lo, hi = v.getOwnershipRange()
            vals = field[indices[:, 1], indices[:, 0]]
            v.setArray(vals[lo:hi])
        # f.dat.data[:] = field[indices[:, 1], indices[:, 0]]
        return f

u_o = gridded_to_point_set(velocity_data["vx"])
v_o = gridded_to_point_set(velocity_data["vy"])
σ_x = gridded_to_point_set(velocity_data["ex"])
σ_y = gridded_to_point_set(velocity_data["ey"])

# Set up and solve a data assimilation problem to interpolate the sparse
# velocity data to the velocity space. This step is necessary because the
# gridded data are missing some points near the terminus
u = firedrake.Function(V0)
N = Constant(len(indices))
area = assemble(Constant(1) * ds_b(mesh))

def loss_functional(u):
    u_int = firedrake.Function(Δ).interpolate(u[0])
    v_int = firedrake.Function(Δ).interpolate(u[1])
    δu = u_int - u_o
    δv = v_int - v_o
    return 0.5 / N * ((δu / σ_x)**2 + (δv / σ_y)**2) * dx

def regularization(u):
    Ω = Constant(area)
    α = Constant(80.0)
    return 0.5 * α**2 / Ω * inner(grad(u), grad(u)) * ds_b

problem = icepack.statistics.StatisticsProblem(
    simulation=lambda u: u.copy(deepcopy=True),
    loss_functional=loss_functional,
    regularization=regularization,
    controls=u,
)

estimator = icepack.statistics.MaximumProbabilityEstimator(
    problem,
    gradient_tolerance=5e-5,
    step_tolerance=1e-8,
    max_iterations=50,
)

u = estimator.solve()

PETSc.Sys.Print("done interpolating velocity data")

# Make an initial estimate for the basal friction by assuming it supports some
# fraction of the driving stress
τ = firedrake.project(-ρ_I * g * h * grad(s), V)

area = assemble(Constant(1) * dx(mesh))
u_avg = assemble(sqrt(inner(u, u)) * dx) / area
τ_avg = assemble(sqrt(inner(τ, τ)) * dx) / area

frac = Constant(0.25)
C = frac * sqrt(inner(τ, τ)) / sqrt(inner(u, u)) ** (1 / m)

def bed_friction(**kwargs):
    u, q = map(kwargs.get, ("velocity", "log_friction"))
    C = Constant(τ_avg) / Constant(u_avg) ** (1 / m) * exp(-q)
    return icepack.models.friction.bed_friction(velocity=u, friction=C)

# Compute an initial estimate for the ice velocity
T = firedrake.Constant(260.0)
A = icepack.rate_factor(T)

flow_model = icepack.models.hybrid.HybridModel(friction=bed_friction)
opts = {
    "dirichlet_ids": [1, 2, 3, 4],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtontr",
        "snes_max_it": 100,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

opts = {
    "dirichlet_ids": [1, 2, 3, 4],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        # "snes_monitor": None,
        "snes_type": "newtonls",
        "snes_linesearch_type": "nleqerr",
        "snes_max_it": 5000,
        "snes_stol": 1.0e-8,
        "snes_rtol": 1.0e-8,
        "snes_atol": 1.0e-3,
        "ksp_type": "bcgs",
        "ksp_max_it": 100000,
        "ksp_rtol": 1.0e-16,
        "ksp_atol": 1.0e-16,
        "pc_type": "bjacobi",
        "pc_hypre_type": "boomeramg",
        "pc_factor_mat_solver_type": "mumps",
        "pc_factor_shift_amount": 1.0e-10,
    },
}
flow_solver = icepack.solvers.FlowSolver(flow_model, **opts)
u_init = firedrake.Function(V).interpolate(u)
u = flow_solver.diagnostic_solve(
    velocity=u_init,
    thickness=h,
    surface=s,
    fluidity=A,
    log_friction=q,
)

PETSc.Sys.Print("done solving for initial velocity")

def simulation(q):
    fields = {"velocity": u_init, "thickness": h, "surface": s, "fluidity": A}
    return flow_solver.diagnostic_solve(log_friction=q, **fields)

def regularization(q):
    Ω = Constant(area)
    α = Constant(args.regularization)
    return 0.5 * α**2 / Ω * inner(grad(q), grad(q)) * ds_b

# Solve a statistical estimation problem for the log-friction
problem = icepack.statistics.StatisticsProblem(
    simulation=simulation,
    loss_functional=loss_functional,
    regularization=regularization,
    controls=q,
)

estimator = icepack.statistics.MaximumProbabilityEstimator(
    problem,
    gradient_tolerance=5e-5,
    step_tolerance=1e-8,
    max_iterations=50,
)

q = estimator.solve()

PETSc.Sys.Print("Doing final solve to get velocity")
u = flow_solver.diagnostic_solve(
    velocity=u_init,
    thickness=h,
    surface=s,
    fluidity=A,
    log_friction=q,
)

C = firedrake.Function(Q).interpolate(Constant(τ_avg) / Constant(u_avg) ** (1 / m) * exp(-q))
# Save the result to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_function(q, name="log_friction")
    chk.save_function(u, name="velocity")
    chk.save_function(u_init, name="velocity_initial")
    chk.save_function(h, name="thickness")
    chk.save_function(s, name="surface")
    chk.save_function(C, name="friction_coefficient")
    chk.h5pyfile.attrs["mean_stress"] = τ_avg
    chk.h5pyfile.attrs["mean_speed"] = u_avg
    chk.h5pyfile.attrs["A"] = float(A)