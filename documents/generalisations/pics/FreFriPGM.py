from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft
# Colors.
integrate_color = {"ec": "#bbbbbb"}
optimize_color = {"ec": "#f89406"}


#------------------------------------------------------------------

# Instantiate the PGM.
pgm = daft.PGM([3.3, 2.55], origin=[0.3, 0.3])

# parameters.
pgm.add_node(daft.Node("MB", r"\begin{center}model for \\background,\\ $B$\end{center}", 1, 2, fixed=True))
pgm.add_node(daft.Node("MC", r"\begin{center}model for \\source,\\ $C$\end{center}", 3, 2, fixed=True))

# Latent variable.
pgm.add_node(daft.Node("theta", r"\begin{center}source \\region, $\theta$ \end{center}", 2, 1.5, aspect=2.2, plot_params=optimize_color))

# Data.
pgm.add_node(daft.Node("nB", r"$\mathbf{n}^B$", 1, 1, observed=True))
pgm.add_node(daft.Node("nC", r"$\mathbf{n}^C$", 3, 1, observed=True))

# Add in the edges.
pgm.add_edge("MB", "nB")
pgm.add_edge("MC", "nC")
pgm.add_edge("theta", "nB")
pgm.add_edge("theta", "nC")

# And a plate.
#pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("FreFriPGM_Schematic.png", dpi=150)

#------------------------------------------------------------------

# Instantiate the PGM.
pgm = daft.PGM([3.3, 3.05], origin=[0.3, 0.3])

# parameters.
pgm.add_node(daft.Node("pB", r"$\mathbf{p}^B$", 1, 2, fixed=True))
pgm.add_node(daft.Node("pC", r"$\mathbf{p}^C$", 3, 2, fixed=True))
pgm.add_node(daft.Node("theta_prior", r"$\beta$", 2, 2, fixed=True))

# Latent variable.
pgm.add_node(daft.Node("theta", r"$\theta$", 2, 1.5, plot_params=optimize_color))

# Data.
pgm.add_node(daft.Node("nB", r"$\mathbf{n}^B$", 1, 1, observed=True))
pgm.add_node(daft.Node("nC", r"$\mathbf{n}^C$", 3, 1, observed=True))

# Add in the edges.
pgm.add_edge("pB", "nB")
pgm.add_edge("pC", "nC")
pgm.add_edge("theta_prior", "theta")
pgm.add_edge("theta", "nB")
pgm.add_edge("theta", "nC")

# And a plate.
#pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("FreFriPGM_Categoricals.png", dpi=150)

#--------------------------------------------------------------

# Instantiate the PGM.
pgm = daft.PGM([3.3, 3.05], origin=[0.3, 0.3])

# Hierarchical parameters.
pgm.add_node(daft.Node("alpha_B", r"$\alpha^B$", 1, 3, fixed=True))
pgm.add_node(daft.Node("alpha_C", r"$\alpha^C$", 3, 3, fixed=True))
pgm.add_node(daft.Node("theta_prior", r"$\beta$", 2, 2, fixed=True))

# Latent variable.
pgm.add_node(daft.Node("pB", r"$\mathbf{p}^B$", 1, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("pC", r"$\mathbf{p}^C$", 3, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("theta", r"$\theta$", 2, 1.5, plot_params=optimize_color))

# Data.
pgm.add_node(daft.Node("nB", r"$\mathbf{n}^B$", 1, 1, observed=True))
pgm.add_node(daft.Node("nC", r"$\mathbf{n}^C$", 3, 1, observed=True))

# Add in the edges.
pgm.add_edge("alpha_B", "pB")
pgm.add_edge("alpha_C", "pC")
pgm.add_edge("pB", "nB")
pgm.add_edge("pC", "nC")
pgm.add_edge("theta_prior", "theta")
pgm.add_edge("theta", "nB")
pgm.add_edge("theta", "nC")

# And a plate.
#pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("FreFriPGM.pdf")
pgm.figure.savefig("FreFriPGM_DirMults.png", dpi=150)


