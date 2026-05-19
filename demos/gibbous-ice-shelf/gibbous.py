import argparse
import numpy as np
import tqdm
import firedrake
from firedrake import (
    inner,
    grad,
    dx,
    ds,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
import irksome
from icepack2 import model
from icepack2.constants import glen_flow_law as n
import gibbous_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--resolution", type=float, default=5e3)
parser.add_argument("--final-time", type=float, default=400.0)
parser.add_argument("--num-steps", type=int, default=200)
parser.add_argument("--calving-freq", type=float, default=0.0)
parser.add_argument("--output", default="steady-state.h5")
args = parser.parse_args()

# Generate and load the mesh
R = 200e3
δx = args.resolution

mesh = gibbous_inputs.make_mesh(R, δx)
h_expr, u_expr = gibbous_inputs.make_initial_data(mesh, R)

# Create some function spaces and some fields
cg = firedrake.FiniteElement("CG", "triangle", 1)
dg0 = firedrake.FiniteElement("DG", "triangle", 0)
dg1 = firedrake.FiniteElement("DG", "triangle", 1)
Q = firedrake.FunctionSpace(mesh, dg1)
V = firedrake.VectorFunctionSpace(mesh, cg)
Σ = firedrake.TensorFunctionSpace(mesh, dg0, symmetry=True)
Z = V * Σ

h0 = firedrake.Function(Q).interpolate(h_expr)
u0 = firedrake.Function(V).interpolate(u_expr)

h = h0.copy(deepcopy=True)
z = firedrake.Function(Z)

# Create some physical constants. Rather than specify the fluidity factor `A`
# in Glen's flow law, we instead define it in terms of a stress scale `τ_c` and
# a strain rate scale `ε_c` as
#
#     A = ε_c / τ_c ** n
#
# where `n` is the Glen flow law exponent. This way, we can do a continuation-
# type method in `n` while preserving dimensional correctness.
ε_c = firedrake.Constant(0.01)
τ_c = firedrake.Constant(0.1)

# If the we passed in some input data, project the inputs into our fn spaces
if args.input:
    with firedrake.CheckpointFile(args.input, "r") as chk:
        input_mesh = chk.load_mesh()
        idx = len(chk.h5pyfile["timesteps"]) - 1
        u_input = chk.load_function(input_mesh, "velocity", idx=idx)
        M_input = chk.load_function(input_mesh, "membrane_stress", idx=idx)
        h_input = chk.load_function(input_mesh, "thickness", idx=idx)

    u_projected = firedrake.project(u_input, V)
    #M_projected = firedrake.project(M_input, Σ)
    h_projected = firedrake.project(h_input, Q)

    z.sub(0).assign(u_projected)
    #z.sub(1).assign(M_projected)
    h0.assign(h_projected)
    h.assign(h_projected)
else:
    z.sub(0).assign(u0)

# Set up the diagnostic problem
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

# Make an initial guess by solving a Picard linearization
linear_rheology = {
    "flow_law_exponent": 1,
    "flow_law_coefficient": ε_c / τ_c,
}

L_1 = sum(fn(**fields, **linear_rheology) for fn in fns)
F_1 = derivative(L_1, z)

qdegree = n + 2
bc = firedrake.DirichletBC(Z.sub(0), u0, (1,))
problem_params = {
    #"form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bc,
}
solver_params = {
    "solver_parameters": {
        "snes_type": "newtonls",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-2,
    },
}
firedrake.solve(F_1 == 0, z, **problem_params, **solver_params)

# Create an approximate linearization
h_min = firedrake.Constant(1.0)
rfields = {
    "velocity": u,
    "membrane_stress": M,
    "thickness": firedrake.max_value(h_min, h),
}

L_r = sum(fn(**rfields, **rheology) for fn in fns)
F_r = derivative(L_r, z)
J_r = derivative(F_r, z)

# Set up the diagnostic problem and solver
# TODO: possibly use a regularized Jacobian
diagnostic_problem = NonlinearVariationalProblem(F, z, J=J_r, **problem_params)
diagnostic_solver = NonlinearVariationalSolver(diagnostic_problem, **solver_params)
diagnostic_solver.solve()

# Set up the prognostic problem and solver
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

# Set up a calving mask
R = firedrake.Constant(60e3)
x = firedrake.SpatialCoordinate(mesh)
y = firedrake.Constant((0.0, R))
mask = firedrake.conditional(inner(x - y, x - y) < R**2, 0.0, 1.0)

time_since_calving = 0.0

# Run the simulation and write the complete output to disk
with firedrake.CheckpointFile(args.output, "w") as chk:
    u, M = z.subfunctions
    chk.save_function(u, name="velocity", idx=0)
    chk.save_function(M, name="membrane_stress", idx=0)
    chk.save_function(h, name="thickness", idx=0)

    for step in tqdm.trange(args.num_steps):
        prognostic_solver.advance()

        if args.calving_freq != 0.0:
            if time_since_calving > args.calving_freq:
                time_since_calving = 0.0
                h.interpolate(mask * h)
            time_since_calving += float(dt)

        h.interpolate(firedrake.max_value(0, h))

        diagnostic_solver.solve()

        u, M = z.subfunctions
        chk.save_function(u, name="velocity", idx=step + 1)
        chk.save_function(M, name="membrane_stress", idx=step + 1)
        chk.save_function(h, name="thickness", idx=step + 1)

    timesteps = np.linspace(0.0, args.final_time, args.num_steps + 1)
    chk.h5pyfile.create_dataset("timesteps", data=timesteps)
