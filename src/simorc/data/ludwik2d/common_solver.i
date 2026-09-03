[Preconditioning]
    [SMP]
        type = SMP
        full = true
    []
[]

[Executioner]
    type = Transient

    solve_type = 'NEWTON'
    petsc_options = '-snes_converged_reason'
    petsc_options_iname = '-pc_type -ksp_type -ksp_gmres_restart -pc_factor_mat_ordering_type'
    petsc_options_value = ' lu       gmres     30                         nd'

    l_max_its = 100
    l_tol = 1e-6

    nl_max_its = 50
    nl_rel_tol = 1e-6
    nl_abs_tol = 1e-6

    end_time = ${endTime}
    dt = 0.5
    dtmin = 1e-5
    dtmax = ${timeStep}

    [TimeStepper]
        type = IterationAdaptiveDT
        dt = 0.5
        optimal_iterations = 8
        iteration_window = 2
        growth_factor = 1.2
        cutback_factor = 0.5
    []

    [Predictor]
        type = SimplePredictor
        scale = 1
    []
[]
