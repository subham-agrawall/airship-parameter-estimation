# airship-parameter-estimation
Nonlinear parameter estimation of airship using modular neural network

Estimated stability and control derivatives of an airship in a completely nonlinear environment. A complete six degrees of freedom airship model has its aerodynamic model as nonlinear functions of angle of attack. Estimating the parameters of aerodynamic model in a nonlinear environment is challenging as it demands an exhaustive dataset that could cover the entire regime of operation of airship. In this work, data generation is achieved by simulating the mathematical model of airship for different trim conditions obtained from continuation analysis. The mathematical model is simulated using predicted parameter values obtained using DATCOM methodology. A modular neural network is then trained using back-propagation and Adam optimisation algorithm for each of the aerodynamic coefficients separately. The estimated nonlinear airship parameters are found to be consistent with the DATCOM parameter values which were used for open-loop simulation. This validates the methodology.

### Journal
The Aeronautical Journal , Volume 124 , Issue 1273 , March 2020 , pp. 409 - 428

DOI: https://doi.org/10.1017/aer.2019.125
