import argparse
import tqdm
from icepack.models.viscosity import Q_warm
import numpy as np
from numpy import pi as π
from ufl.classes import ufl_classes
import xarray
import firedrake
from firedrake import assemble, Constant, exp, max_value, inner, dx, ds_v, dS_v, min_value
import icepack
from icepack.calculus import grad
from icepack2.constants import (
    glen_flow_law,
    weertman_sliding_law,
    ice_density as ρ_I,
    water_density as ρ_W,
)
from icepack2.model import hybrid
from icepack2 import model
import irksome

TEST = True
CUSTOMH = True

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="kangerdlugssuaq-initial-bp.h5")
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--snes-max-it", type=int, default=250)
parser.add_argument("--snes-rtol", type=float, default=1e-8)
parser.add_argument("--output", default="kangerdlugssuaq-check-bp.h5")
parser.add_argument("--vdegree", type=int, default=2)
args = parser.parse_args()

# Read in the starting data
with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh("kangerdlugssuaq")
    q = chk.load_function(mesh, name="log_friction")
    u = chk.load_function(mesh, name="velocity")
    h = chk.load_function(mesh, name="thickness")
    s_in = chk.load_function(mesh, name="surface")
    C_in = chk.load_function(mesh, name="friction_coefficient")
    τ_c = chk.h5pyfile.attrs["mean_stress"]
    u_c = chk.h5pyfile.attrs["mean_speed"]
    A = firedrake.Constant(chk.h5pyfile.attrs["A"])

if args.vdegree == 0:
    vfamily = "R"
else:
    vfamily = "DG"

Q = firedrake.FunctionSpace(mesh, "CG", args.degree, vfamily="R", vdegree=0)
Δ = firedrake.FunctionSpace(mesh, "DG", args.degree, vfamily="R", vdegree=0)
V = firedrake.VectorFunctionSpace(mesh, "CG", args.degree, dim=2, vfamily=vfamily, vdegree=args.vdegree)
Σx = firedrake.TensorFunctionSpace(mesh, "DG", args.degree - 1, shape=(2, 2), symmetry=True, vfamily=vfamily, vdegree=max(0, args.vdegree - 1))
T = firedrake.VectorFunctionSpace(mesh, "DG", args.degree - 1, dim=2, vfamily=vfamily, vdegree=max(0, args.vdegree - 1))
T0 = firedrake.VectorFunctionSpace(mesh, "DG", args.degree - 1, dim=2, vfamily="R", vdegree=0)
Z = V * Σx * T * T0

h = firedrake.Function(Q).interpolate(h)
q = firedrake.Function(Q).interpolate(q)

u = firedrake.Function(V).interpolate(u)
u_in = u.copy(deepcopy=True)

z = firedrake.Function(Z)
z.sub(0).assign(u_in)

# Read the thickness and bed data
bedmachine = xarray.open_dataset(icepack.datasets.fetch_bedmachine_greenland())
b = icepack.interpolate(bedmachine["bed"], Q)
h = firedrake.project(h, Δ)
s = firedrake.project(s_in, Δ)

rheology_steps = 5
ms = np.linspace(1.0, weertman_sliding_law, rheology_steps)
ns = np.linspace(1.0, glen_flow_law, rheology_steps)

m_slide = firedrake.Constant(ms[0])
n_flow = firedrake.Constant(ns[0])
n_flow.assign(glen_flow_law)
m_slide.assign(weertman_sliding_law)

# Set up the momentum balance equation and solve
ε_c = Constant(A * τ_c ** glen_flow_law)
print(f"τ_c: {1000 * float(τ_c):.1f} kPa")
print(f"ε_c: {1000 * float(ε_c):.1f} (m / yr) / km")
print(f"u_c: {float(u_c):.1f} m / yr")

u, Mx, Mz, τ = firedrake.split(z)
fields = {
    "velocity": u,
    "membrane_stress_x": Mx,
    "membrane_stress_z": Mz,
    "basal_stress": τ,
    "thickness": h,
    "surface": s,
}

h_min = Constant(1.0)
rfields = {
    "velocity": u,
    "membrane_stress_x": Mx,
    "membrane_stress_z": Mz,
    "basal_stress": τ,
    "thickness": max_value(h_min, h),
    "surface": s,
}

# bedfric_high = firedrake.Constant(-10.0)
# q = firedrake.Function(Q).interpolate(bedfric_high)
rheology = {
    "flow_law_exponent": n_flow,
    "flow_law_coefficient": ε_c / τ_c**n_flow,
    # "flow_law_exponent": 1,
    # "flow_law_coefficient": ε_c / τ_c,
    "sliding_exponent": m_slide,
    "sliding_coefficient": u_c / τ_c**m_slide * exp(m_slide * q),
}

linear_rheology = {
    "flow_law_exponent": 1,
    "flow_law_coefficient": ε_c / τ_c,
    "sliding_exponent": 1,
    "sliding_coefficient": u_c / τ_c * exp(q),
}

# L_1 = hybrid.HybridModel(calving_terminus=None).action(**fields, **linear_rheology)
# F_1 = firedrake.derivative(L_1, z)
# J_1 = firedrake.derivative(F_1, z)
# L_r = hybrid.HybridModel(calving_terminus=None).action(**rfields, **rheology)
# F_r = firedrake.derivative(L_r, z)
# J_r = firedrake.derivative(F_r, z)
# J = J_r + α * J_1

L = hybrid.HybridModel(calving_terminus=None).action(**fields, **rheology)
F = firedrake.derivative(L, z)

inflow_ids = [1, 2, 3, 4]
bc_in = firedrake.DirichletBC(Z.sub(0), u_in, inflow_ids)
bcs = [bc_in]

qdegree = int(max(weertman_sliding_law, glen_flow_law)) + 2
problem_params = {
    "form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bcs,
}
solver_params = {
    "solver_parameters": {
        #"snes_linesearch_monitor": None,
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
        "pc_factor_mat_solver_type": "mumps",
    },
}
solver_params["solver_parameters"]["snes_monitor"] = None
# solver_params["solver_parameters"]["ksp_monitor"] = None
# firedrake.solve(F_1 == 0, z, **problem_params, **solver_params)

u_problem = firedrake.NonlinearVariationalProblem(F, z, **problem_params)  # may need to add J=J here
u_solver = firedrake.NonlinearVariationalSolver(u_problem, **solver_params)
for rheo_step in range(rheology_steps):
    m_slide.assign(ms[rheo_step])
    n_flow.assign(ns[rheo_step])
    u_solver.solve()

def bed_friction(**kwargs):
    u, q = map(kwargs.get, ("velocity", "log_friction"))
    C = Constant(τ_c) / Constant(u_c) ** (1 / weertman_sliding_law) * exp(-q)
    return icepack.models.friction.bed_friction(velocity=u, friction=C)

flow_model = icepack.models.hybrid.HybridModel(friction=bed_friction)
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
u_primal = firedrake.Function(V).interpolate(u)
u_primal = flow_solver.diagnostic_solve(
    velocity=u_primal,
    thickness=h,
    surface=s,
    fluidity=A,
    log_friction=q,
    flow_law_exponent=firedrake.Constant(glen_flow_law),
)

K = firedrake.Function(Q).interpolate(u_c / τ_c**m_slide * exp(m_slide * q))

u, Mx, Mz, τ = z.subfunctions

with firedrake.CheckpointFile(args.output, "w") as chk:
    chk.save_mesh(mesh)
    chk.save_function(u, name="velocity_dual")
    chk.save_function(u_primal, name="velocity_primal")
    chk.save_function(Mx, name="membrane_stress_x")
    chk.save_function(Mz, name="membrane_stress_z")
    chk.save_function(τ, name="basal_stress")
    chk.save_function(q, name="log_friction")
    chk.save_function(K, name="sliding_coefficient")
    chk.h5pyfile.attrs["mean_stress"] = τ_c
    chk.h5pyfile.attrs["mean_speed"] = u_c