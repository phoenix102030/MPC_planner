import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Define the time step and horizon
dt = 0.1  # time step
N = 20    # prediction horizon
T = 50    # total number of simulation steps

# Define vehicle parameters
L = 2.0   # wheelbase
v = 0.1   # constant velocity

# Define reference trajectory (circle for example)
t = np.linspace(0, T*dt, T+1)
X_ref = np.vstack((np.cos(t), np.sin(t), t))

# Define CasADi variables
X = ca.SX.sym('X', 3, N+1)  # state variables [X, Y, theta]
delta = ca.SX.sym('delta', N)  # control inputs (steering angle)
X_0 = ca.SX.sym('X_0', 3)  # initial state

# Define the cost function and constraints
cost = 0
constraints = []

for k in range(N):
    # Cost function
    cost += ca.sumsqr(X[:, k] - X_ref[:, k]) + ca.sumsqr(delta[k])

    # System dynamics
    next_state = X[:, k] + dt * ca.vertcat(
        v * ca.cos(X[2, k]),
        v * ca.sin(X[2, k]),
        v / L * ca.tan(delta[k])
    )
    constraints.append(X[:, k+1] - next_state)

# Initial state constraint
constraints.append(X[:, 0] - X_0)

# Combine all constraints into a single vector
constraints = ca.vertcat(*constraints)

# Set up the optimization problem
opt_variables = ca.vertcat(ca.reshape(X, -1, 1), delta)
nlp = {'x': opt_variables, 'f': cost, 'g': constraints, 'p': X_0}
opts = {
    'ipopt.print_level': 0,
    'ipopt.max_iter': 100,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-4,
}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Initial state
X_0_val = X_ref[:, 0]

# Storage for simulation data
X_sim = [X_0_val]
delta_sim = []

# MPC loop
for i in range(T):
    # Initial guess
    X_guess = np.tile(X_0_val, (N+1, 1)).T
    delta_guess = np.zeros(N)
    initial_guess = np.concatenate([X_guess.flatten(), delta_guess])

    # Solve the optimization problem
    solution = solver(
        x0=initial_guess,
        p=X_0_val,
        lbg=-ca.inf,
        ubg=ca.inf,
        lbx=-ca.inf,
        ubx=ca.inf,
    )

    # Extract the optimal solution
    X_opt = np.transpose(np.array(solution['x'][:3*(N+1)]).reshape(N+1, 3))
    delta_opt = np.array(solution['x'][3*(N+1):])

    # Apply the first control input
    delta_applied = delta_opt[0]
    delta_sim.append(delta_applied)

    # Update the state
    X_0_val = X_opt[:, 1]

    # Store the state
    X_sim.append(X_0_val)

# Convert simulation data to numpy arrays
X_sim = np.array(X_sim).T
delta_sim = np.array(delta_sim)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(X_ref[0, :], X_ref[1, :], 'r--', label='Reference Trajectory')
plt.plot(X_sim[0, :], X_sim[1, :], 'b-', label='Optimized Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('MPC for Kinematic Bicycle Model with CasADi')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(delta_sim, 'b-', label='Steering Angle (delta)')
plt.xlabel('Time Step')
plt.ylabel('Steering Angle (rad)')
plt.title('Optimal Steering Angle Over Time')
plt.legend()
plt.show()
