*KEYWORD
*INCLUDE
fine.inc
*ALE_MULTI-MATERIAL_GROUP
         1         1
         2         1
*CONTROL_ALE
         0         1         2      -1.0

*CONTROL_TERMINATION
     100.0
*DATABASE_BINARY_D3PLOT
       1.0 
*DATABASE_GLSTAT
       0.1
*INITIAL_DETONATION
         1 0.000E+00 0.000E+00 0.000E+00 0.000E+00
*CONTROL_ENERGY
         2         1         2         2
*MAT_NULL
         1       1.0
*EOS_GRUNEISEN
         1       1.5

*INITIAL_DETONATION
         2
*MAT_HIGH_EXPLOSIVE_BURN
         2      1.63     0.784     0.260 0.000E+00
*EOS_JWL
         2      3.71 3.230E-02      4.15     0.950     0.300 4.300E-02      1.00
*SECTION_SOLID
         1        11
*SECTION_SOLID
         2        11 
*PART

         1         1         1         1
*PART

         2         2         2         2
*END
