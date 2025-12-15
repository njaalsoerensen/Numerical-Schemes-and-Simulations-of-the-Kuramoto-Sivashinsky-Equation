Code usage:

Audun_solver.py is A solver for KS equation written by Audun Theodorsen.

Audun_solver_FS.py is a Audun_solver.py rewritten to be able to be compared with the numerical methods dissused in the project.

KS_compute.py contains all the code code used to generate plots.
Sectoins of code in __name__ == '__main__' block is organized within HTML style sections <section></section> of commentout out code.
 - '<Solve history>' is used to generatte contour plots of solutions.
 - '''<Energy conservation conformation>''' is used to generate plots relating to the energy conservation equation, the neergy phase space, and the energy over time.
 - '''<Confirm behavour>''' Is used to generate animations of solutions for all methods in time domain, the methods used can be found without the run_full_ETD() function.
 - '''<Confirm behavour FT>''' is the same as confirm behavour, but for the solutions given in frequency domain.
 - '''<Confirm behavour FS>''' is the same is the same as confirm bahavour and is redundant. Originally used to compare ETD-RK4 and Audun_solver_FS.
 - '''<Linear growth>''' is used to generate plots relating to linear growth of modes in Fourier domain.
 - '''<Testing vectorized version>''' is same as confirm behavour and was used to compare perofrmance of vectorized and uvecotrized versions of solvers and is redundant.
 - '''<Error relationship>''' Is used to generate plots relating the the error of variable frequency and spatial resolution as well as the runtime.
 - '''<Average energy>''' is same as Energy conservation conformation.

KS_schemes.py contains classes of solvers and buils on the base KS_equation in KS_base.

KS_Base.py contains the base KS equation with discritized points as well as methods to plot initial curves and do timesteps, appending solutiosn, call a solver, etc.. KS_base also contains the Bank class which can organize multiple KS equations as well as ploting multiple solutions of multiple KS equations, computing error with MSE between two solutions, and ploting contour plots.

kuramoto_sivashinsky_helper.oy is a wirten by Audun Theodorsen and is used to run kuramto_sivashinsky_sim.py

kuramoto_sivashinsky_sim.py is written by Audun Theoderson and used to run the KS solver written by Audun Theoderson and can be used to plot contour plots of solutions and the energy of the system.
