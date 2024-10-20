import numpy as np

# fundamental quad parameters
quad_params = {
    "mB": 1.2, # mass (kg)
    "dxm": 0.16, # arm length (m)
    "dym": 0.16, # arm length (m)
    "dzm": 0.01, # arm height (m)
    "IB": np.array([[0.0123, 0,      0     ],
                    [0,      0.0123, 0     ],
                    [0,      0,      0.0224]]), # Inertial tensor (kg*m^2)
    "IRzz": 2.7e-5, # rotor moment of inertia (kg*m^2)
    "Cd": 0.1, # drag coefficient (omnidirectional)
    "kTh": 1.076e-5, # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
    "kTo": 1.632e-7, # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
    "minThr": 0.1*4, # Minimum total thrust (N)
    "maxThr": 9.18*4, # Maximum total thrust (N)
    "minWmotor": 75, # Minimum motor rotation speed (rad/s)
    "maxWmotor": 925, # Maximum motor rotation speed (rad/s)
    "tau": 0.015, # Value for second order system for Motor dynamics
    "kp": 1.0, # Value for second order system for Motor dynamics
    "damp": 1.0, # Value for second order system for Motor dynamics
    "usePrecession": True, # model precession or not
    # "w_hover": 522.9847140714692, # hardcoded hover rotor speed (rad/s)
}

# post init useful parameters for quad
quad_params["B0"] = np.array([
    [quad_params["kTh"], quad_params["kTh"], quad_params["kTh"], quad_params["kTh"]],
    [quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], -quad_params["dym"]*quad_params["kTh"], quad_params["dym"]*quad_params["kTh"]],
    [quad_params["dxm"]*quad_params["kTh"], quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"], -quad_params["dxm"]*quad_params["kTh"]],
    [-quad_params["kTo"], quad_params["kTo"], -quad_params["kTo"], quad_params["kTo"]]]) # actuation matrix

quad_params["x_lb"] = np.array([
    *[-10]*3, *[-np.inf]*4, *[-10]*3, *[-10]*3, *[quad_params["minWmotor"]]*4
    # xyz       q0123         xdydzd    pdqdrd    w0123
])

quad_params["x_ub"] = np.array([
    *[10]*3, *[np.inf]*4, *[10]*3, *[10]*3, *[quad_params["maxWmotor"]]*4
    # xyz      q0123        xdydzd   pdqdrd   w0123
])


# Fundamental Simulation Parameters
sim_params = {
    "Ti": 0.,
    "Tf": 30.,
    "Ts": 0.05 # 0.001, # 0.01, ~0.05 also works
}


# Fundamental Feedback Linearization Parameters (includes attitude control parameters)
ctrl_params = {
    "Lambda": np.diag([1.0, 1.0, 1.0]),  # Tuning parameter for position error
    "K": np.diag([2.0, 2.0, 2.0]),       # Gain matrix for s
    "K_R": 1.0 * np.eye(3),              # Attitude control gain
    "K_omega": 0.1 * np.eye(3),          # Angular velocity control gain
    "K_p_w": 0.001 / sim_params["Ts"] * 0.05, # Rotor proportional gain
    "tilde_p_lb": np.array([-100.]*3),
    "tilde_p_ub": np.array([100.]*3)
}




