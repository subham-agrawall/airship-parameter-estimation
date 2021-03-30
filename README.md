# Nonlinear parameter estimation of airship using modular neural network

Estimated stability and control derivatives of an airship in a completely nonlinear environment. A complete six degrees of freedom airship model has its aerodynamic model as nonlinear functions of angle of attack. Estimating the parameters of aerodynamic model in a nonlinear environment is challenging as it demands an exhaustive dataset that could cover the entire regime of operation of airship. In this work, data generation is achieved by simulating the mathematical model of airship for different trim conditions obtained from continuation analysis. The mathematical model is simulated using predicted parameter values obtained using DATCOM methodology. A modular neural network is then trained using back-propagation and Adam optimisation algorithm for each of the aerodynamic coefficients separately. The estimated nonlinear airship parameters are found to be consistent with the DATCOM parameter values which were used for open-loop simulation. This validates the methodology.

Full details can be found in this article - The Aeronautical Journal (DOI: https://doi.org/10.1017/aer.2019.125)

## Code-flow:
1. constants.py - Airship paramaters (geometric and other data)
2. generate_trim.py - Trim points passed an initial states for simulation
3. airship.py - Airship simulation
4. generate_data.py - Runs simulation for each trim point by calling airship.py. Simulation data is then saved as csv.
5. solve_linear.py (optional) 
6. cd.py cl.py cm.py cn.py cr.py cy.py - Parameter estimation using Modular Neural Network
