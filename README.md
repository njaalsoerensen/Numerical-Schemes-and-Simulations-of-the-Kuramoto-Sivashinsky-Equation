Code usage:

KS_env.yaml is the environment used to run all the code used in this project. This enviornment contains python3 and the minimal packages required to run the code.

KS_Base.py contains the base KS_equation class with discretized points as well as methods to plot initial curves and do timestep, appending solutions, call a solver, etc.. KS_base also contains the Bank class which can organize multiple KS equations as well as plotting multiple solutions of multiple KS equations, computing error with MSE between two solutions, and piloting contour plots.

KS_Schemes.py contains classes of solvers, this is where the numerical schemes are described. KS_schemes builds on the the base class KS_equation in KS_base.

KS_Compute.py contains all the code code used to do computations relation to the project and is used to generate .png plots as well as .gif animations. Sections of code in name == 'main' block is organized within HTML style sections <\section> of commented out code.

RUN_*.py files are used to generate plots described by its name. These can be used to either run for one specific hyperviscosity value, or it can run for all possible states at once.

Audun_solver.py is A solver for KS equation written by Audun Theodorsen.

Audun.py is a Audun_solver.py rewritten to be able to be compared with the numerical ETD-RK methods discussed in the project.

kuramoto_sivashinsky_helper.py is a written by Audun Theodorsen and is used to run kuramto_sivashinsky_sim.py.

kuramoto_sivashinsky_sim.py is written by Audun Theoderson and used to run the KS solver written by Audun Theoderson.
