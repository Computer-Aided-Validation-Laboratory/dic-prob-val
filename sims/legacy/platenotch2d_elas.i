#-------------------------------------------------------------------------
# pyvale: gmsh,mechanical,transient
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START
!include common_load_time.i
!include common_elas_props.i
simName = notch2d_elas
#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[Variables]
    [scalar_strain_zz]
    []
[]


[GlobalParams]
    displacements = 'disp_x disp_y'
    out_of_plane_strain = scalar_strain_zz
[]

[Mesh]
    type = FileMesh
    file = 'mesh2d_notchplate.msh'
[]

[Physics/SolidMechanics/QuasiStatic]
    [all]
        strain = SMALL
        planar_formulation = WEAK_PLANE_STRESS
        incremental = true
        add_variables = true
        material_output_family = MONOMIAL
        material_output_order = CONSTANT
        # generate_output = 'vonmises_stress strain_xx strain_yy strain_zz strain_xy'
        generate_output = 'vonmises_stress strain_xx strain_yy strain_zz strain_xy stress_xx stress_yy stress_zz stress_xy'
    []
[]

[BCs]
    [bottom_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-bot-mid'
        value = 0.0
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = 'bc-bot'
        value = 0.0
    []

    [top_y]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'bc-top'
        function = '${topDispRate}*t'
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${EMod}
        poissons_ratio = ${PRatio}
    []
    [stress]
        type = ComputeFiniteStrainElasticStress
    []
[]

!include common_solver.i


[Postprocessors]
    [react_y_top]
        type = SidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'bc-top'
    []
    [disp_y_max]
        type = NodalExtremeValue
        variable = disp_y
    []
[]

!include common_outputs.i
