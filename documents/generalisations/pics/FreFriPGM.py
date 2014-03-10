from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft
# Colors.
integrate_color = {"ec": "#bbbbbb"}
optimize_color = {"ec": "#f89406"}


# Instantiate the PGM.
pgm = daft.PGM([4.3, 3.05], origin=[0.3, 0.3])

# Hierarchical parameters.
pgm.add_node(daft.Node("alpha_0", r"$\alpha_0$", 1, 3, fixed=True))
pgm.add_node(daft.Node("alpha_1", r"$\alpha_1$", 3, 3, fixed=True))
pgm.add_node(daft.Node("theta_prior", r"$\beta$", 2, 3, fixed=True))

# Latent variable.
pgm.add_node(daft.Node("P_0", r"$P_0$", 1, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("P_1", r"$P_1$", 3, 2, plot_params=integrate_color))
pgm.add_node(daft.Node("theta", r"$\theta$", 2, 2, plot_params=optimize_color))

# Data.
pgm.add_node(daft.Node("n_0", r"$n_0$", 1, 1, observed=True))
pgm.add_node(daft.Node("n_1", r"$n_1$", 3, 1, observed=True))

# Add in the edges.
pgm.add_edge("alpha_0", "P_0")
pgm.add_edge("alpha_1", "P_1")
pgm.add_edge("P_0", "n_0")
pgm.add_edge("P_1", "n_1")
pgm.add_edge("theta_prior", "theta")
pgm.add_edge("theta", "n_0")
pgm.add_edge("theta", "n_1")

# And a plate.
#pgm.add_plate(daft.Plate([0.5, 0.5, 2, 1], label=r"$n = 1, \cdots, N$",shift=-0.1))

# Render and save.
pgm.render()
pgm.figure.savefig("FreFriPGM.pdf")
pgm.figure.savefig("FreFriPGM.png", dpi=150)
