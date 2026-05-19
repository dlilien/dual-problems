import json
import argparse
import numpy as np
from numpy import pi as π
import tqdm
import firedrake
from firedrake import inner, derivative
import irksome
import icepack
from icepack2 import model
from icepack2.constants import glen_flow_law as n

parser = argparse.ArgumentParser()
parser.add_argument("--form", choices=["primal", "dual"])
parser.add_argument("--hmin", type=float, default=1e-3)
parser.add_argument("--rtol", type=float, default=1e-4)
parser.add_argument("--final-time", type=float, default=400.0)
parser.add_argument("--num-steps", type=int, default=400)
parser.add_argument("--calving-frequency", type=float, default=24.0)
parser.add_argument("--output")
args = parser.parse_args()

# Load the input data
with firedrake.CheckpointFile("steady-state-coarse.h5", "r") as chk:
    mesh = chk.load_mesh()
    idx = len(chk.h5pyfile["timesteps"]) - 1
    h = chk.load_function(mesh, "thickness", idx=idx)
    h0 = h.copy(deepcopy=True)
    u = chk.load_function(mesh, "velocity", idx=idx)
    u_0 = u.copy(deepcopy=True)
    M = chk.load_function(mesh, "membrane_stress", idx=idx)

Q = h.function_space()
V = u.function_space()

# Set up the Lagrangian
ε_c = firedrake.Constant(0.01)
τ_c = firedrake.Constant(0.1)
h_min = firedrake.Constant(args.hmin)

params = {
    "snes_type": "newtonls",
    "snes_rtol": args.rtol,
    "snes_linesearch_type": "nleqerr",
    "snes_monitor": None,
}

# Set up the diagnostic solver and boundary conditions and do an initial solve
if args.form == "primal":
    from icepack.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
    from firedrake import div
    def gravity(**kwargs):
        u = kwargs["velocity"]
        h = kwargs["thickness"]
        ρ = ρ_I * (1 - ρ_I / ρ_W)
        return 0.5 * ρ * g * h**2 * div(u)

    def terminus(**kwargs):
        return firedrake.Constant(0.0)

    _model = icepack.models.IceShelf(gravity=gravity, terminus=terminus)

    u = u.copy(deepcopy=True)
    A = firedrake.Constant(ε_c / τ_c ** n)
    opts = {
        "diagnostic_solver_type": "petsc",
        "diagnostic_solver_parameters": params,
    }
    outer_solver = icepack.solvers.FlowSolver(_model, dirichlet_ids=[1], **opts)
    outer_solver.diagnostic_solve(velocity=u, thickness=h, fluidity=A)
    h = outer_solver.fields["thickness"]
    diagnostic_solver = outer_solver._diagnostic_solver._solver

elif args.form == "dual":
    Σ = M.function_space()
    Z = V * Σ
    z = firedrake.Function(Z)
    z.sub(0).assign(u)
    z.sub(1).assign(M);

    u, M = firedrake.split(z)
    fields = {
        "velocity": u,
        "membrane_stress": M,
        "thickness": h,
    }

    fns = [model.minimization.viscous_power, model.minimization.ice_shelf_momentum_balance]

    rheology = {
        "flow_law_exponent": n,
        "flow_law_coefficient": ε_c / τ_c ** n,
    }

    L = sum(fn(**fields, **rheology) for fn in fns)
    F = derivative(L, z)

    # Set up a regularized Lagrangian
    rfields = {
        "velocity": u,
        "membrane_stress": M,
        "thickness": firedrake.max_value(h_min, h),
    }

    L_r = sum(fn(**rfields, **rheology) for fn in fns)
    F_r = derivative(L_r, z)
    J_r = derivative(F_r, z)

    # Set up the diagnostic solver and boundary conditions and do an initial solve
    bc = firedrake.DirichletBC(Z.sub(0), u_0, (1,))
    diagnostic_problem = firedrake.NonlinearVariationalProblem(F, z, J=J_r, bcs=bc)
    params = {"solver_parameters": params}
    diagnostic_solver = firedrake.NonlinearVariationalSolver(diagnostic_problem, **params)
    diagnostic_solver.solve()


# Set up the mass balance equation
prognostic_problem = model.mass_balance(
    thickness=h,
    velocity=u,
    accumulation=firedrake.Constant(0.0),
    thickness_inflow=h0,
    test_function=firedrake.TestFunction(Q),
)

dt = firedrake.Constant(args.final_time / args.num_steps)
t = firedrake.Constant(0.0)
method = irksome.BackwardEuler()
prognostic_params = {
    "solver_parameters": {
        "snes_type": "ksponly",
        "ksp_type": "gmres",
        "pc_type": "bjacobi",
    },
}
prognostic_solver = irksome.TimeStepper(
    prognostic_problem, method, t, dt, h, **prognostic_params
)


# Create the calving mask -- this describes how we'll remove ice
radius = firedrake.Constant(60e3)
x = firedrake.SpatialCoordinate(mesh)
y = firedrake.Constant((0.0, radius))
mask = firedrake.conditional(inner(x - y, x - y) < radius**2, 0.0, 1.0)

# Run the simulation
time_since_calving = 0.0

num_newton_iterations = []
for step in range(args.num_steps):
    prognostic_solver.advance()

    if time_since_calving > args.calving_frequency:
        h.interpolate(mask * h)
        time_since_calving = 0.0
    time_since_calving += float(dt)
    expr = firedrake.max_value(0 if args.form == "dual" else h_min, h)
    h.interpolate(expr)
    diagnostic_solver.solve()
    num_newton_iterations.append(diagnostic_solver.snes.getIterationNumber())


# Save the results to disk
with open(args.output, "w") as output_file:
    json.dump(num_newton_iterations, output_file)
