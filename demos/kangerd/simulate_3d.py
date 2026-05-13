import argparse
import subprocess
from icepack.models.viscosity import Q_warm
import numpy as np
from numpy import pi as π
import xarray
import firedrake
from firedrake import assemble, Constant, exp, max_value, inner, dx, ds_v, dS_v
import icepack
from icepack.calculus import grad
from icepack2.constants import (
    glen_flow_law as n,
    weertman_sliding_law as m,
    ice_density as ρ_I,
    water_density as ρ_W,
)
from icepack2 import model

TEST = True

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="kangerdlugssuaq-extrapolated.h5")
parser.add_argument("--timesteps-per-year", type=int, default=192)
parser.add_argument("--final-time", type=float, default=1)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--calving", action="store_true")
parser.add_argument("--melt-rate", type=float, default=3e3)
parser.add_argument("--mask-smoothing-length", type=float, default=5e3)
parser.add_argument("--snes-max-it", type=int, default=200)
parser.add_argument("--snes-rtol", type=float, default=1e-6)
parser.add_argument("--output", default="kangerdlugssuaq-year1-3d.h5")
args = parser.parse_args()

# Read in the starting data
with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh2d = chk.load_mesh()
    q2d = chk.load_function(mesh2d, name="log_friction")
    τ_c = chk.h5pyfile.attrs["mean_stress"]
    u_c = chk.h5pyfile.attrs["mean_speed"]

    timesteps = np.array(chk.h5pyfile["timesteps"])
    u2d = chk.load_function(mesh2d, name="velocity", idx=len(timesteps) - 1)
    h2d = chk.load_function(mesh2d, name="thickness", idx=len(timesteps) - 1)
mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)


Q = firedrake.FunctionSpace(mesh, "CG", args.degree, vfamily="R", vdegree=0)
Δ = firedrake.FunctionSpace(mesh, "DG", args.degree, vfamily="R", vdegree=0)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree, dim=2, vfamily="R", vdegree=0)
Σ = firedrake.TensorFunctionSpace(mesh, "DG", args.degree - 1, shape=(2, 2), symmetry=True, vfamily="R", vdegree=0)
T = firedrake.VectorFunctionSpace(mesh, "DG", args.degree - 1, dim=2, vfamily="R", vdegree=0)
Z = V * Σ * T

if TEST:
    Q2d = firedrake.FunctionSpace(mesh2d, "CG", args.degree)
    Δ2d = firedrake.FunctionSpace(mesh2d, "DG", args.degree)
    V2d = firedrake.VectorFunctionSpace(mesh2d, "CG", args.degree)
    Σ2d = firedrake.TensorFunctionSpace(mesh2d, "DG", args.degree - 1, symmetry=True)
    T2d = firedrake.VectorFunctionSpace(mesh2d, "DG", args.degree - 1)

    for fs3d, fs2d in zip([Q, Δ, V, Σ, T], [Q2d, Δ2d, V2d, Σ2d, T2d]):
        qs3d = firedrake.Function(fs3d)
        qs2d = firedrake.Function(fs2d)
        assert qs3d.dat.data_ro.shape == qs2d.dat.data_ro.shape

u = icepack.lift3d(u2d, V)
h = icepack.lift3d(h2d, Q)
q = icepack.lift3d(q2d, Q)

u_in = firedrake.project(u, V)
q = firedrake.project(q, Q)

z = firedrake.Function(Z)
z.sub(0).assign(u_in)

# Read the thickness and bed data
bedmachine = xarray.open_dataset(icepack.datasets.fetch_bedmachine_greenland())
b = icepack.interpolate(bedmachine["bed"], Q)
h = firedrake.project(h, Δ)
s = firedrake.project(max_value(b + h, (1 - ρ_I / ρ_W) * h), Δ)

# Set up the momentum balance equation and solve
A = icepack.rate_factor(Constant(260))
ε_c = Constant(A * τ_c ** n)
print(f"τ_c: {1000 * float(τ_c):.1f} kPa")
print(f"ε_c: {1000 * float(ε_c):.1f} (m / yr) / km")
print(f"u_c: {float(u_c):.1f} m / yr")

fns = [
    model.minimization.viscous_power,
    model.minimization.friction_power,
    #model.calving_terminus,
    model.minimization.momentum_balance,
]

u, M, τ = firedrake.split(z)
fields = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": h,
    "surface": s,
}

h_min = Constant(1.0)
rfields = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": max_value(h_min, h),
    "surface": s,
}

rheology = {
    "flow_law_exponent": n,
    "flow_law_coefficient": ε_c / τ_c**n,
    "sliding_exponent": m,
    "sliding_coefficient": u_c / τ_c**m * exp(m * q),
}

linear_rheology = {
    "flow_law_exponent": 1,
    "flow_law_coefficient": ε_c / τ_c,
    "sliding_exponent": 1,
    "sliding_coefficient": u_c / τ_c * exp(q),
}

L_1 = sum(fn(**rfields, **linear_rheology) for fn in fns)
F_1 = firedrake.derivative(L_1, z)
J_1 = firedrake.derivative(F_1, z)

L = sum(fn(**fields, **rheology) for fn in fns)
F = firedrake.derivative(L, z)

L_r = sum(fn(**rfields, **rheology) for fn in fns)
F_r = firedrake.derivative(L_r, z)
J_r = firedrake.derivative(F_r, z)
α = firedrake.Constant(0.01)
J = J_r + α * J_1

inflow_ids = [1]
bc_in = firedrake.DirichletBC(Z.sub(0), u_in, inflow_ids)
outflow_ids = [2, 3, 4]
bc_out = firedrake.DirichletBC(Z.sub(0), Constant((0.0, 0.0)), outflow_ids)
bcs = [bc_in, bc_out]

qdegree = int(max(m, n)) + 2
problem_params = {
    "form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bcs,
}
solver_params = {
    "solver_parameters": {
        "snes_monitor": None,
        #"snes_converged_reason": None,
        #"ksp_monitor": None,
        #"ksp_view": None,
        "snes_stol": 0.0,
        "snes_rtol": args.snes_rtol,
        "snes_max_it": args.snes_max_it,
        "snes_divergence_tolerance": -1,
        "snes_type": "newtonls",
        "snes_linesearch_type": "nleqerr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "umfpack",
    },
}
#firedrake.solve(F_1 == 0, z, **problem_params, **solver_params)

u_problem = firedrake.NonlinearVariationalProblem(F, z, J=J, **problem_params)
u_solver = firedrake.NonlinearVariationalSolver(u_problem, **solver_params)
u_solver.solve()

α.assign(0.0)

# Fix the accumulation rate. We used estimates of surface mass balance from the
# regional climate model MAR and remote sensing measurements of surface
# elevation to estimate a linear relationship:
#
#     SMB ~= da_ds * s + a_0
#
# where `da_ds` ~= 2.25 milimeters of water equivalent per year per meter
# elevation gain and `a_0` ~= -3.3 meters of water equivalent per year at sea
# level. We used all the data from 2006-2017 and the fit had `r² = 0.91`.
# See also https://www.climato.uliege.be/cms/c_5652668/fr/climato-greenland.
da_ds = Constant(2.25 * 1e-3)
a_0 = Constant(-3.3)
smb = 0.917 * (a_0 + da_ds * s)

# Create a solver for a smooth ice mask field
if args.calving:
    α = Constant(args.mask_smoothing_length)
    μ = firedrake.Function(Q)
    χ = firedrake.conditional(h > 0, 1, 0)
    J = 0.5 * ((μ - χ)**2 + α**2 * inner(grad(μ), grad(μ))) * dx
    bcs = [
        firedrake.DirichletBC(Q, Constant(1.0), (1,)),
        firedrake.DirichletBC(Q, Constant(0.0), (3,)),
    ]
    F = firedrake.derivative(J, μ)
    μ_problem = firedrake.NonlinearVariationalProblem(F, μ, bcs)
    μ_solver = firedrake.NonlinearVariationalSolver(μ_problem)
    μ_solver.solve()

# Create the accumulation/ablation function, which is a sum of SMB and calving
# losses (if any)
t = Constant(0.0)
if args.calving:
    m = Constant(args.melt_rate)  # melt rate in m/yr
    φ = firedrake.min_value(0, firedrake.cos(2 * π * t))
    calving = m * (1 - μ) * φ
    a = smb + calving
else:
    a = smb

# Set up the mass balance equation
h_n = h.copy(deepcopy=True)
h0 = h.copy(deepcopy=True)
φ = firedrake.TestFunction(h.function_space())
dt = Constant(1.0 / args.timesteps_per_year)
flux_cells = ((h - h_n) / dt * φ - inner(h * u, grad(φ)) - a * φ) * dx
ν = firedrake.FacetNormal(mesh)
f = h * max_value(0, u[0] * ν[0] + u[1] * ν[1])
flux_facets = (f("+") - f("-")) * (φ("+") - φ("-")) * dS_v
flux_in = h0 * firedrake.min_value(0, u[0] * ν[0] + u[1] * ν[1]) * φ * ds_v
flux_out = h * max_value(0, u[0] * ν[0] + u[1] * ν[1]) * φ * ds_v
G = flux_cells + flux_facets + flux_in + flux_out
h_problem = firedrake.NonlinearVariationalProblem(G, h)
h_solver = firedrake.NonlinearVariationalSolver(h_problem)

# Run the simulation
h_c = Constant(5.0)
num_steps = int(args.final_time * args.timesteps_per_year) + 1
with firedrake.CheckpointFile(args.output, "w") as chk:
    u, M, τ = z.subfunctions
    chk.save_function(h, name="thickness", idx=0)
    chk.save_function(s, name="surface", idx=0)
    chk.save_function(u, name="velocity", idx=0)
    chk.save_function(M, name="membrane_stress", idx=0)
    chk.save_function(τ, name="basal_stress", idx=0)
    if args.calving:
        chk.save_function(μ, name="ice_mask", idx=0)

    timesteps = np.linspace(0.0, args.final_time, num_steps)
    for step in range(num_steps):
        t.assign(t + dt)
        if args.calving:
            μ_solver.solve()

        h_solver.solve()
        h.interpolate(firedrake.conditional(h < h_c, 0, h))
        h_n.assign(h)
        s.interpolate(max_value(b + h, (1 - ρ_I / ρ_W) * h))
        u_solver.solve()

        # Save the results to disk
        u, M, τ = z.subfunctions
        chk.save_function(h, name="thickness", idx=step + 1)
        chk.save_function(s, name="surface", idx=step + 1)
        chk.save_function(u, name="velocity", idx=step + 1)
        chk.save_function(M, name="membrane_stress", idx=step + 1)
        chk.save_function(τ, name="basal_stress", idx=step + 1)
        if args.calving:
            chk.save_function(μ, name="ice_mask", idx=step + 1)

    chk.save_function(q, name="log_friction")
    chk.h5pyfile.attrs["mean_stress"] = τ_c
    chk.h5pyfile.attrs["mean_speed"] = u_c
    chk.h5pyfile.create_dataset("timesteps", data=timesteps)
