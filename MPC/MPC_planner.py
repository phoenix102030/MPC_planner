import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the time step and horizon
dt = 0.1  # time step
N = 20    # prediction horizon
T = 100    # total number of simulation steps

# Define vehicle parameters
L = 2.0   # wheelbase
v = 1.0   # constant velocity

# Define reference trajectory (circle for example)
t = np.linspace(0, T*dt, T+1)
X_ref = np.vstack((t, np.sin(0.5*t), 0.1 * t))

# Define CasADi variables
X = ca.SX.sym('X', 3, N+1)  # state variables [X, Y, theta]
delta = ca.SX.sym('delta', N)  # control inputs (steering angle)
X_0 = ca.SX.sym('X_0', 3)  # initial state

# Define the cost function and constraints
cost = 0
constraints = []

# for k in range(N):
#     # Cost function
#     cost += ca.sumsqr(X[:, k] - X_ref[:, k]) + ca.sumsqr(delta[k])

#     # System dynamics
#     next_state = X[:, k] + dt * ca.vertcat(
#         v * ca.cos(X[2, k]),
#         v * ca.sin(X[2, k]),
#         v / L * ca.tan(delta[k])
#     )
#     constraints.append(X[:, k+1] - next_state)

# # Initial state constraint
# constraints.append(X[:, 0] - X_0)

# # Combine all constraints into a single vector
# constraints = ca.vertcat(*constraints)

# Set up the optimization problem
opt_variables = ca.vertcat(ca.reshape(X, -1, 1), delta)
nlp = {'x': opt_variables, 'f': cost, 'g': constraints, 'p': X_0}
opts = {
    'ipopt.print_level': 0,
    'ipopt.max_iter': 100,
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-4,
}
# solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# Initial state
X_0_val = X_ref[:, 0]

# Storage for simulation data
X_sim = [X_0_val]
delta_sim = []

# Ser steering angel constraints
delta_min = -np.pi/4
delta_max = np.pi/4

# Prepare figure for animation
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X_ref[0, :], X_ref[1, :], 'r--', label='Reference Trajectory')
actual_line, = ax.plot([], [], 'b-', label='Actual Trajectory')
prediction_line, = ax.plot([], [], 'g-', label='Predicted Trajectory')
ax.set_xlim(0, T*dt)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.set_title('MPC for Kinematic Bicycle Model with CasADi')

def update(frame):
    global X_0_val

    # Adjust prediction horizon if remaining steps are less than N
    current_N = min(N, T - frame)

    # Redefine CasADi variables for the current horizon
    X = ca.SX.sym('X', 3, current_N + 1)
    delta = ca.SX.sym('delta', current_N)
    cost = 0
    constraints = []

    for k in range(current_N):
        cost += 100 * ca.sumsqr(X[:, k] - X_ref[:, frame + k]) #+ ca.sumsqr(delta[k])
        next_state = X[:, k] + dt * ca.vertcat(
            v * ca.cos(X[2, k]),
            v * ca.sin(X[2, k]),
            v / L * ca.tan(delta[k])
        )
        constraints.append(X[:, k+1] - next_state)

    constraints.append(X[:, 0] - X_0)

    constraints = ca.vertcat(*constraints)
    opt_variables = ca.vertcat(ca.reshape(X, -1, 1), delta)
    nlp = {'x': opt_variables, 'f': cost, 'g': constraints, 'p': X_0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Initial guess
    X_guess = np.tile(X_0_val, (current_N+1, 1)).T
    delta_guess = np.zeros(current_N)
    initial_guess = np.concatenate([X_guess.flatten(), delta_guess])

    lbx = np.concatenate([-ca.inf * np.ones((3 * (current_N + 1))),
                         delta_min * np.ones(current_N)])
    ubx = np.concatenate([ca.inf * np.ones((3 * (current_N + 1))),
                         delta_max * np.ones(current_N)])

    # Solve the optimization problem
    solution = solver(
        x0=initial_guess,
        p=X_0_val,
        lbg=0,
        ubg=0,
        lbx=lbx,
        ubx=ubx,
    )

    # Extract the optimal solution
    X_opt = np.transpose(np.array(solution['x'][:3*(current_N+1)]).reshape(current_N+1, 3))
    delta_opt = np.array(solution['x'][3*(current_N+1):])

    # Apply the first control input
    delta_applied = delta_opt[0]
    delta_sim.append(delta_applied)

    # Update the state
    X_0_val = X_opt[:, 1]

    # Store the state
    X_sim.append(X_0_val)

    # Update the plot
    actual_line.set_data(np.array(X_sim).T[0, :], np.array(X_sim).T[1, :])
    prediction_line.set_data(X_opt[0, :], X_opt[1, :])
    return actual_line, prediction_line

ani = animation.FuncAnimation(fig, update, frames=T, blit=True, repeat=False)
ani.save('mpc_simulation.gif', writer='imagemagick')

plt.show()
