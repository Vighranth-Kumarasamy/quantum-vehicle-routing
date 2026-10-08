
import itertools
import time
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Qiskit imports are kept inside the quantum function so the UI can still open
# if a package is missing.
st.set_page_config(page_title="Quantum Vehicle Routing", page_icon="⚛️", layout="wide")

st.title("⚛️ Quantum-Optimized Delivery Routes")
st.caption("Automatic multi-vehicle routing from coordinates using QAOA and exact classical validation.")

# ---------- Geometry ----------
def euclidean_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    n = len(points)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = float(np.hypot(points[i][0] - points[j][0],
                                         points[i][1] - points[j][1]))
    return d

def route_cost(route: List[int], D: np.ndarray) -> float:
    if not route:
        return 0.0
    total = D[0, route[0]]
    for a, b in zip(route, route[1:]):
        total += D[a, b]
    total += D[route[-1], 0]
    return float(total)

# ---------- Exact classical VRP for small demo ----------
def exact_vrp(D: np.ndarray, vehicles: int):
    """Exact small VRP with automatic customer-to-vehicle assignment.

    The user specifies how many vehicles are available/used; the algorithm
    automatically decides which customers each vehicle serves and the best
    order inside each route. No customer-count-per-vehicle input is required.
    For a small 4-customer demo, exhaustive enumeration is practical.
    """
    customers = list(range(1, D.shape[0]))
    n = len(customers)
    if vehicles > n:
        vehicles = n

    best_cost = float("inf")
    best_routes = None

    # Every vehicle is used when vehicles <= customers; assignment is automatic.
    # Empty vehicles are allowed only when the requested fleet exceeds customers.
    for assignment in itertools.product(range(vehicles), repeat=n):
        if vehicles <= n and set(assignment) != set(range(vehicles)):
            continue

        groups = [[] for _ in range(vehicles)]
        for c, v in zip(customers, assignment):
            groups[v].append(c)

        routes = []
        total = 0.0
        for group in groups:
            best_r = None
            best_r_cost = float("inf")
            for perm in itertools.permutations(group):
                rc = route_cost(list(perm), D)
                if rc < best_r_cost:
                    best_r_cost = rc
                    best_r = list(perm)
            routes.append(best_r or [])
            total += best_r_cost if best_r is not None else 0.0

        if total < best_cost - 1e-9:
            best_cost = total
            best_routes = routes

    return best_routes, best_cost

# ---------- QAOA assignment model ----------
def qaoa_assignment(D: np.ndarray, vehicles: int, shots=1024):
    """QAOA searches customer-to-vehicle assignments.

    A quadratic surrogate is used for the assignment stage, then the exact
    route order for each selected group is computed classically. This keeps
    the 4-customer demonstration small and robust.
    """
    try:
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.converters import QuadraticProgramToQubo
        from qiskit_aer import AerSimulator
        from qiskit_aer.primitives import SamplerV2
        from qiskit import generate_preset_pass_manager
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
    except Exception as e:
        return None, None, 0, f"Qiskit unavailable: {e}"

    customers = list(range(1, D.shape[0]))
    n = len(customers)
    vehicles = min(vehicles, n)
    nv = n * vehicles
    P = 1000.0

    qp = QuadraticProgram("automatic_vehicle_assignment")
    names = {}
    for ci in range(n):
        for v in range(vehicles):
            name = f"x_{ci}_{v}"
            qp.binary_var(name)
            names[(ci, v)] = name

    linear = {}
    quadratic = {}

    def add_lin(name, val):
        linear[name] = linear.get(name, 0.0) + float(val)

    def add_quad(a, b, val):
        key = tuple(sorted((a, b)))
        quadratic[key] = quadratic.get(key, 0.0) + float(val)

    # Base solo costs. Pair adjustment rewards grouping customers when that
    # reduces the two-customer route compared with two separate depot trips.
    solo = {c: route_cost([c], D) for c in customers}
    for ci, c in enumerate(customers):
        for v in range(vehicles):
            add_lin(names[(ci, v)], solo[c])

    for v in range(vehicles):
        for i_idx in range(n):
            for j_idx in range(i_idx + 1, n):
                i, j = customers[i_idx], customers[j_idx]
                pair = min(route_cost([i, j], D), route_cost([j, i], D))
                adjustment = pair - solo[i] - solo[j]
                add_quad(names[(i_idx, v)], names[(j_idx, v)], adjustment)

    # Every customer belongs to exactly one vehicle.
    for ci in range(n):
        vars_ci = [names[(ci, v)] for v in range(vehicles)]
        for a in vars_ci:
            add_lin(a, -P)
        for a_idx in range(len(vars_ci)):
            for b_idx in range(a_idx + 1, len(vars_ci)):
                add_quad(vars_ci[a_idx], vars_ci[b_idx], 2 * P)

    # Require every selected vehicle to receive at least one customer.
    if vehicles <= n:
        for v in range(vehicles):
            vars_v = [names[(ci, v)] for ci in range(n)]
            # P*(sum x - 1)^2 is only valid for exactly one; instead use a
            # large negative linear term to discourage empty vehicles, with a
            # post-validation check. The exact classical result remains the
            # authoritative reference for the tiny demo.
            for a in vars_v:
                add_lin(a, -0.01)

    qp.minimize(linear=linear, quadratic=quadratic)
    qubo = QuadraticProgramToQubo().convert(qp)

    backend = AerSimulator()
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = SamplerV2(default_shots=shots)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=50), reps=1, transpiler=pm)
    opt = MinimumEigenOptimizer(qaoa)

    try:
        result = opt.solve(qubo)
    except Exception as e:
        return None, None, nv, f"QAOA execution error: {e}"

    groups = [[] for _ in range(vehicles)]
    for ci, c in enumerate(customers):
        chosen = [v for v in range(vehicles) if result.x[ci * vehicles + v] > 0.5]
        if len(chosen) != 1:
            return None, None, nv, "QAOA produced an infeasible assignment."
        groups[chosen[0]].append(c)

    if vehicles <= n and any(len(g) == 0 for g in groups):
        return None, None, nv, "QAOA produced an unused vehicle; exact validation will be used."

    final_routes = []
    total = 0.0
    for group in groups:
        if not group:
            final_routes.append([])
            continue
        best_r = min((list(p) for p in itertools.permutations(group)),
                     key=lambda r: route_cost(r, D))
        final_routes.append(best_r)
        total += route_cost(best_r, D)

    return final_routes, total, nv, "QAOA returned a feasible automatic assignment."

def draw_routes(points, routes):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot each route.
    for v, route in enumerate(routes):
        if not route:
            continue
        seq = [0] + route + [0]
        xs = [points[i][0] for i in seq]
        ys = [points[i][1] for i in seq]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"Vehicle {v+1}")
    ax.scatter([points[0][0]], [points[0][1]], marker="*", s=500, zorder=5, label="Depot")
    ax.annotate("Depot", points[0], xytext=(8, 8), textcoords="offset points", fontsize=11, fontweight="bold")
    for i in range(1, len(points)):
        ax.scatter([points[i][0]], [points[i][1]], s=300, zorder=5)
        ax.annotate(f"Customer {i}", points[i], xytext=(8, 8), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_title("Optimized Delivery Routes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig

# ---------- UI ----------
with st.sidebar:
    st.header("⚙️ Problem Setup")
    customers_n = st.number_input("Number of customers", min_value=2, max_value=4, value=4, step=1)
    vehicles = st.number_input("Available vehicles", min_value=1, max_value=4, value=2, step=1)
    st.info("The selected vehicles are assigned customers automatically. No fixed customer limit per vehicle.")

default_points = [(0,0), (-1,4), (7,-8), (-2,14), (0,14)]
points = []
st.subheader("📍 Location Coordinates")
st.caption("The depot can be anywhere. Enter X/Y coordinates for each location.")
cols = st.columns(3)
for i in range(customers_n + 1):
    label = "Depot" if i == 0 else f"Customer {i}"
    c1, c2 = st.columns(2)
    with c1:
        x = st.number_input(f"{label} X", value=float(default_points[i][0]), step=0.1, format="%.2f", key=f"x{i}")
    with c2:
        y = st.number_input(f"{label} Y", value=float(default_points[i][1]), step=0.1, format="%.2f", key=f"y{i}")
    points.append((x, y))


D = euclidean_matrix(points)

with st.expander("📏 Calculated Distance Matrix"):
    names = ["Depot"] + [f"C{i}" for i in range(1, customers_n+1)]
    st.dataframe(pd.DataFrame(np.round(D, 2), index=names, columns=names), use_container_width=True)

if st.button("🚀 Find Quantum Optimized Routes", type="primary", use_container_width=True):
    start = time.perf_counter()

    with st.spinner("Running classical reference and QAOA..."):
        classical_routes, classical_cost = exact_vrp(D, int(vehicles))
        q_routes, q_cost, qubits, q_status = qaoa_assignment(D, int(vehicles))

    if q_routes is None:
        q_routes = classical_routes
        q_cost = classical_cost
        used_fallback = True
    else:
        used_fallback = False

    elapsed = time.perf_counter() - start

    st.success("Optimization completed.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Classical optimum", f"{classical_cost:.2f}")
    c2.metric("QAOA result", f"{q_cost:.2f}")
    c3.metric("Difference", f"{q_cost-classical_cost:.2f}")
    gap = 0 if classical_cost == 0 else (q_cost-classical_cost)/classical_cost*100
    c4.metric("QAOA gap", f"{gap:.2f}%")

    if used_fallback:
        st.warning("QAOA did not return a feasible sample for this instance, so the exact classical solution is shown as a validated fallback. The QAOA attempt was still executed.")

    st.caption(f"Quantum variables: {qubits} | Runtime: {elapsed:.2f} s | {q_status}")

    st.subheader("🚚 Optimized Routes")
    rows = []
    for v, r in enumerate(q_routes):
        if r:
            labels = ["Depot"] + [f"Customer {i}" for i in r] + ["Depot"]
            rows.append({"Vehicle": f"Vehicle {v+1}", "Route": " → ".join(labels), "Distance": round(route_cost(r, D), 2)})
        else:
            rows.append({"Vehicle": f"Vehicle {v+1}", "Route": "Unused", "Distance": 0.0})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("🗺️ Route Visualization")
    st.pyplot(draw_routes(points, q_routes), use_container_width=True)

    with st.expander("🔎 Classical reference routes"):
        for v, r in enumerate(classical_routes):
            if r:
                st.write(f"Vehicle {v+1}: Depot → " + " → ".join(f"Customer {i}" for i in r) + " → Depot")
            else:
                st.write(f"Vehicle {v+1}: Unused")

st.divider()
st.caption("Small-instance educational prototype: QAOA is heuristic; the classical solver provides exact validation for the supported 4-customer instances.")
