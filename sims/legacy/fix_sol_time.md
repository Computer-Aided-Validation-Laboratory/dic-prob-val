The major thing was the numerical stiffness of the problem, I added variable scaling to disp_x and disp_y,
    [disp_x]
      order = 2
      scaling = 1e-9
    []
and this helped dramatically in terms of actually converging in each timestep.

I also fiddled a bit with your solver settings, I reduced the GMRES restart and set the matrix ordering explicitly. 
    petsc_options_iname = '-pc_type -ksp_type -ksp_gmres_restart -pc_factor_mat_ordering_type'
    petsc_options_value = ' lu       gmres     30  nd'

For me this went from not solving in 40 minutes to solving in less than 2.
