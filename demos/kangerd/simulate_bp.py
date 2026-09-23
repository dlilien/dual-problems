import argparse
import tqdm
import numpy as np
from numpy import pi as π
import xarray
import firedrake
from firedrake import Constant, exp, max_value, inner, dx, ds_v, dS_v
from firedrake.petsc import PETSc
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
from mpi4py import MPI

# These scales can be used for non-dimensionalization. From Greve 2025.
TAU_SCALE = 0.1 #0.1  # 100 kPa
EPS_SCALE = 0.05 #0.025  # per year


def nondim_A(A, n, τ_c=TAU_SCALE, ε_c=EPS_SCALE):
    """Calculate A_tilde from Greve 2025."""
    return A / (ε_c / τ_c ** n)


TEST = True
CUSTOMH = True

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="kangerdlugssuaq-extrapolated-bp.h5")
parser.add_argument("--timesteps-per-year", type=int, default=192)
parser.add_argument("--final-time", type=float, default=1.0)
parser.add_argument("--degree", type=int, default=1)
parser.add_argument("--calving", action="store_true")
parser.add_argument("--melt-rate", type=float, default=3e3)
parser.add_argument("--mask-smoothing-length", type=float, default=5e3)
parser.add_argument("--snes-max-it", type=int, default=2500)
parser.add_argument("--snes-rtol", type=float, default=1e-6)
parser.add_argument("--output", default="kangerdlugssuaq-year1-bp.h5")
parser.add_argument("--vdegree", type=int, default=2)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

timesteps_per_year = args.timesteps_per_year

# Read in the starting data
with firedrake.CheckpointFile(args.input, "r") as chk:
    mesh = chk.load_mesh("kangerdlugssuaq_enlarged")
    q = chk.load_function(mesh, name="log_friction")
    τ_c = chk.h5pyfile.attrs["mean_stress"]
    u_c = chk.h5pyfile.attrs["mean_speed"]

    timesteps = np.array(chk.h5pyfile["timesteps"])
    u = chk.load_function(mesh, name="velocity", idx=len(timesteps) - 1)
    h = chk.load_function(mesh, name="thickness", idx=len(timesteps) - 1)

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
s = firedrake.Function(Δ).interpolate(max_value(b + h, (1 - ρ_I / ρ_W) * h))

rheology_steps = 5
ms = np.linspace(1.0, weertman_sliding_law, rheology_steps)
ns = np.linspace(1.0, glen_flow_law, rheology_steps)

m_slide = firedrake.Constant(ms[0])
n_flow = firedrake.Constant(ns[0])

# Set up the momentum balance equation and solve
A3 = icepack.rate_factor(Constant(260))
ε_c = Constant(A3 * τ_c ** glen_flow_law)
A_tilde = nondim_A(A3, 3.0, τ_c=τ_c, ε_c=ε_c)

PETSc.Sys.Print(f"τ_c: {1000 * float(τ_c):.1f} kPa")
PETSc.Sys.Print(f"ε_c: {1000 * float(ε_c):.1f} (m / yr) / km")
PETSc.Sys.Print(f"u_c: {float(u_c):.1f} m / yr")

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

rheology = {
    "flow_law_exponent": n_flow,
    "flow_law_coefficient": A_tilde * ε_c / τ_c**n_flow,
    "sliding_exponent": m_slide,
    "sliding_coefficient": u_c / τ_c**m_slide * exp(m_slide * q),
}

linear_rheology = {
    "flow_law_exponent": 1,
    "flow_law_coefficient": A_tilde * EPS_SCALE / TAU_SCALE,
    "sliding_exponent": 1,
    "sliding_coefficient": u_c / τ_c * exp(q),
}

α = Constant(0.01)
L_1 = hybrid.HybridModel(calving_terminus=None).action(**fields, **linear_rheology)
F_1 = firedrake.derivative(L_1, z)
J_1 = firedrake.derivative(F_1, z)
L_r = hybrid.HybridModel(calving_terminus=None).action(**rfields, **rheology)
F_r = firedrake.derivative(L_r, z)
J_r = firedrake.derivative(F_r, z)
J = J_r + α * J_1

L = hybrid.HybridModel(calving_terminus=None).action(**fields, **rheology)
F = firedrake.derivative(L, z)

inflow_ids = [1]
bc_in = firedrake.DirichletBC(Z.sub(0), u_in, inflow_ids)
outflow_ids = [2, 3, 4]
bc_out = firedrake.DirichletBC(Z.sub(0), Constant((0.0, 0.0)), outflow_ids)
bcs = [bc_in, bc_out]

qdegree = int(max(weertman_sliding_law, glen_flow_law)) + 2
problem_params = {
    "form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bcs,
}
solver_params = {
    "solver_parameters": {
        # "snes_linesearch_monitor": None,
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
if args.debug:
    solver_params["solver_parameters"]["snes_monitor"] = None
    solver_params["solver_parameters"]["ksp_monitor"] = None
# firedrake.solve(F_1 == 0, z, **problem_params, **solver_params)

u_problem = firedrake.NonlinearVariationalProblem(F, z, J=J, **problem_params)  # may need to add J=J here
u_solver = firedrake.NonlinearVariationalSolver(u_problem, **solver_params)
for rheo_step in range(rheology_steps):
    m_slide.assign(ms[rheo_step])
    n_flow.assign(ns[rheo_step])
    u_solver.solve()
# α.assign(0.0)
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

h0 = h.copy(deepcopy=True)
dt = Constant(1.0 / timesteps_per_year)


# Set up the mass balance equation
if CUSTOMH:
    h_n = h.copy(deepcopy=True)
    φ = firedrake.TestFunction(h.function_space())
    flux_cells = ((h - h_n) / dt * φ - inner(h * u, grad(φ)) - a * φ) * dx
    ν = firedrake.FacetNormal(mesh)
    f = h * max_value(0, u[0] * ν[0] + u[1] * ν[1])
    flux_facets = (f("+") - f("-")) * (φ("+") - φ("-")) * dS_v
    flux_in = h0 * firedrake.min_value(0, u[0] * ν[0] + u[1] * ν[1]) * φ * ds_v
    flux_out = h * max_value(0, u[0] * ν[0] + u[1] * ν[1]) * φ * ds_v
    G = flux_cells + flux_facets + flux_in + flux_out
    h_problem = firedrake.NonlinearVariationalProblem(G, h)
    h_solver = firedrake.NonlinearVariationalSolver(h_problem)
else:
    # Set up the prognostic problem and solver
    prognostic_problem = model.mass_balance(
        thickness=h,
        velocity=u,
        accumulation=firedrake.Constant(0.0),
        thickness_inflow=h0,
        test_function=firedrake.TestFunction(Q),
    )
    t = firedrake.Constant(0.0)
    method = irksome.BackwardEuler()
    prognostic_params = {
        "solver_parameters": {
            "snes_type": "ksponly",
            "ksp_type": "gmres",
            "pc_type": "bjacobi",
        },
    }
    h_solver = irksome.TimeStepper(
        prognostic_problem, method, t, dt, h, **prognostic_params
    )

# Run the simulation
h_c = Constant(5.0)
num_steps = int(args.final_time * timesteps_per_year) + 1
with firedrake.CheckpointFile(args.output, "w") as chk:
    u, Mx, Mz, τ = z.subfunctions
    chk.save_function(h, name="thickness", idx=0)
    chk.save_function(u, name="velocity", idx=0)
    chk.save_function(s, name="surface", idx=0)
    chk.save_function(Mx, name="membrane_stress_x", idx=0)
    chk.save_function(Mz, name="membrane_stress_z", idx=0)
    chk.save_function(τ, name="basal_stress", idx=0)
    if args.calving:
        chk.save_function(μ, name="ice_mask", idx=0)
    chk.save_function(q, name="log_friction")
    chk.h5pyfile.attrs["mean_stress"] = τ_c
    chk.h5pyfile.attrs["mean_speed"] = u_c
    timesteps = np.linspace(0.0, args.final_time, num_steps)
    chk.h5pyfile.create_dataset("timesteps", data=timesteps)

    for step in tqdm.trange(num_steps, disable=MPI.COMM_WORLD.rank > 0):
        t.assign(t + dt)
        if args.calving:
            μ_solver.solve()

        if CUSTOMH:
            h_solver.solve()
            h_n.assign(h)
        else:
            h_solver.advance()
        # h.interpolate(firedrake.conditional(h < h_c, 0, h))
        s.interpolate(max_value(b + h, (1 - ρ_I / ρ_W) * h))
        try:
            try:
                u_solver.solve()
            except Exception as e:
                for rheo_step in range(rheology_steps):
                    m_slide.assign(ms[rheo_step])
                    n_flow.assign(ns[rheo_step])
                    u_solver.solve()
        except Exception as e:
            raise e
        finally:
            # Save the results to disk
            u, Mx, Mz, τ = z.subfunctions
            chk.save_function(h, name="thickness", idx=step + 1)
            chk.save_function(s, name="surface", idx=step + 1)
            chk.save_function(u, name="velocity", idx=step + 1)
            chk.save_function(Mx, name="membrane_stress_x", idx=step + 1)
            chk.save_function(Mz, name="membrane_stress_z", idx=step + 1)
            chk.save_function(τ, name="basal_stress", idx=step + 1)
            if args.calving:
                chk.save_function(μ, name="ice_mask", idx=step + 1)