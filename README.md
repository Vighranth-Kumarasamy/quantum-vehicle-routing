# Quantum Vehicle Routing - Final Working Model

## Final behavior
- 2-4 customers
- 1-4 vehicles
- The selected vehicle count is the fleet used for the routing problem.
- Customers are assigned automatically; there is NO "maximum customers per vehicle" input.
- One vehicle may serve multiple customers.
- Customer coordinates use decimal values (0.1 step, 2 decimal display).
- Depot and every customer can be placed anywhere.
- Exact classical solver provides the small-instance reference.
- QAOA searches the customer-to-vehicle assignment; route order is optimized exactly for the selected assignment.
- If QAOA does not return a feasible assignment, the exact classical solution is shown as a clearly labeled fallback.

## Run
1. Install Python 3.11.
2. Extract this folder.
3. Double-click `run_app.bat`.
4. Open the localhost address shown in the terminal.

## Recommended demo
- Customers: 4
- Vehicles: 2
- Coordinates can be entered as decimals, e.g. 2.50, 3.75.

Note: With a pure total-distance objective and no capacity/time constraints, the optimal assignment depends on the actual coordinates. The application does not force a particular customer grouping; it calculates the lowest-cost valid assignment automatically.
