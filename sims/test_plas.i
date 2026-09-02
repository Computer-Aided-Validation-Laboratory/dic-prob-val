#-------------------------------------------------------------------------
# pyvale: gmsh,mechanical,transient
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START
!include common_load_time.i
!include common_plas_linear_props.i
simName = test_plas

plate_width = 25.0
c1_bot = 0.0
c2_bot = 0.0
c1_top = 0.0
c2_top = 0.0
#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[Variables]
    [disp_x]
        order = SECOND
        scaling = 1e-9
    []
    [disp_y]
        order = SECOND
        scaling = 1e-9
    []
    [scalar_strain_zz]
        scaling = 1e-9
    []
[]

[GlobalParams]
    displacements = 'disp_x disp_y'
    out_of_plane_strain = scalar_strain_zz
[]

[Mesh]
    type = FileMesh
    file = 'hole_notch_2d.msh'
[]

[Functions]
    [bc_bot_y_fn]
        type = ParsedFunction
        expression = '(${c1_bot} * x + ${c2_bot} * (x^2 - (${plate_width}^2)/12)) * (t / ${endTime})'
    []
    [bc_top_y_fn]
        type = ParsedFunction
        expression = '(${topDispRate} * t) + (${c1_top} * x + ${c2_top} * (x^2 - (${plate_width}^2)/12)) * (t / ${endTime})'
    []
[]

[Physics/SolidMechanics/QuasiStatic]
    [all]
        strain = FINITE
        planar_formulation = WEAK_PLANE_STRESS
        incremental = true
        add_variables = false
        use_automatic_differentiation = true

        material_output_family = MONOMIAL
        material_output_order = CONSTANT
        generate_output = 'vonmises_stress strain_xx strain_yy strain_zz strain_xy plastic_strain_xx plastic_strain_yy plastic_strain_zz plastic_strain_xy stress_xx stress_yy stress_zz stress_xy'
    []
[]

[Materials]
    [elasticity]
        type = ADComputeIsotropicElasticityTensor
        youngs_modulus = ${EMod}
        poissons_ratio = ${PRatio}
    []

    [radial_return_stress]
        type = ADComputeMultipleInelasticStress
        inelastic_models = 'isoplas'
    []

    [isoplas]
        type = ADIsotropicPlasticityStressUpdate
        yield_stress = ${Yield}
        hardening_constant = ${HardMod}
        relative_tolerance = 1e-9
        absolute_tolerance = 1e-9
    []
[]

[AuxVariables]
    [effective_plastic_strain_out]
        family = MONOMIAL
        order = CONSTANT
    []
[]

[AuxKernels]
    [effective_plastic_strain_out]
        type = ADMaterialRealAux
        variable = effective_plastic_strain_out
        property = effective_plastic_strain
        execute_on = 'INITIAL TIMESTEP_END'
    []
[]

[BCs]
    [bottom_x]
        type = ADDirichletBC
        variable = disp_x
        boundary = 'bc-bot-mid'
        value = 0.0
    []
    [bottom_y]
        type = ADFunctionDirichletBC
        variable = disp_y
        boundary = 'bc-bot'
        function = bc_bot_y_fn
    []

    [top_y]
        type = ADFunctionDirichletBC
        variable = disp_y
        boundary = 'bc-top'
        function = bc_top_y_fn
    []
[]

!include common_solver.i

[Postprocessors]
    [react_y_top]
        type = ADSidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'bc-top'
    []

    [disp_y_max]
        type = NodalExtremeValue
        variable = disp_y
    []

    [stress_vm_max]
        type = ElementExtremeValue
        variable = vonmises_stress
    []

    [plastic_strain_eq_max]
        type = ElementExtremeValue
        variable = effective_plastic_strain_out
    []
[]

!include common_outputs.i
