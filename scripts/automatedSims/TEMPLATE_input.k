*KEYWORD
*INCLUDE
mesh.inc
*ALE_MULTI-MATERIAL_GROUP
         1         1
         2         1
*CONTROL_ALE
         0         1         2      -1.0

*CONTROL_TERMINATION
      70.0
*DATABASE_BINARY_D3PLOT
       1.0
*DATABASE_GLSTAT
       0.1
*DATABASE_BINARY_RUNRSF
        10
*DATABASE_ELOUT
       0.5
*DATABASE_NODOUT
       0.5
*DATABASE_HISTORY_SOLID_SET
      7001
*DATABASE_HISTORY_NODE_SET
      7101
*INITIAL_DETONATION
         1 0.000E+00 0.000E+00 0.000E+00 0.000E+00
*CONTROL_ENERGY
         2         1         2         2
*MAT_NULL
         1       1.0
*EOS_MURNAGHAN
       1        7.15  3.31E-03 9.999E-01

*INITIAL_DETONATION
         2
*MAT_HIGH_EXPLOSIVE_BURN
         2      1.63     0.693      0.21 0.000E+00
*EOS_JWL
         2     3.712 3.230E-02      4.15     0.950     0.300  4.29E-02      1.00
*SECTION_SOLID
         1        11
*SECTION_SOLID
         2        11
*PART

         1         1         1         1
*PART

         2         2         2         2
*SENSOR_DEFINE_ELEMENT_SET
      9002     SOLID      9001        XX    STRESS
       MIN
*SENSOR_SWITCH
      9003                9002        LT    -0.002
*TERMINATION_SENSOR
      9003
*END
