import numpy as np
from .config import E1, E2, Gamma1, Gamma2

# Material 1 Lamé constants
Lambda1 = E1 * Gamma1 / ((1 + Gamma1) * (1 - 2 * Gamma1))
Mu1 = E1 / (2 * (1 + Gamma1))

# Material 2 Lamé constants
Lambda2 = E2 * Gamma2 / ((1 + Gamma2) * (1 - 2 * Gamma2))
Mu2 = E2 / (2 * (1 + Gamma2))

R11_mat1 = Lambda1 + 2 * Mu1
R12_mat1 = Lambda1
R13_mat1 = Lambda1
R14_mat1 = 0
R15_mat1 = 0
R16_mat1 = 0

R21_mat1 = R12_mat1
R22_mat1 = Lambda1 + 2 * Mu1
R23_mat1 = Lambda1
R24_mat1 = 0
R25_mat1 = 0
R26_mat1 = 0

R31_mat1 = R13_mat1
R32_mat1 = R23_mat1
R33_mat1 = Lambda1 + 2 * Mu1
R34_mat1 = 0
R35_mat1 = 0
R36_mat1 = 0

R41_mat1 = 0
R42_mat1 = 0
R43_mat1 = 0
R44_mat1 = Mu1
R45_mat1 = 0
R46_mat1 = 0

R51_mat1 = 0
R52_mat1 = 0
R53_mat1 = 0
R54_mat1 = 0
R55_mat1 = Mu1
R56_mat1 = 0

R61_mat1 = 0
R62_mat1 = 0
R63_mat1 = 0
R64_mat1 = 0
R65_mat1 = 0
R66_mat1 = Mu1

R_mat1 = np.array([
    [R11_mat1, R12_mat1, R13_mat1, R14_mat1, R15_mat1, R16_mat1],
    [R21_mat1, R22_mat1, R23_mat1, R24_mat1, R25_mat1, R26_mat1],
    [R31_mat1, R32_mat1, R33_mat1, R34_mat1, R35_mat1, R36_mat1],
    [R41_mat1, R42_mat1, R43_mat1, R44_mat1, R45_mat1, R46_mat1],
    [R51_mat1, R52_mat1, R53_mat1, R54_mat1, R55_mat1, R56_mat1],
    [R61_mat1, R62_mat1, R63_mat1, R64_mat1, R65_mat1, R66_mat1],
])

R11_mat2 = Lambda2 + 2 * Mu2
R12_mat2 = Lambda2
R13_mat2 = Lambda2
R14_mat2 = 0
R15_mat2 = 0
R16_mat2 = 0

R21_mat2 = R12_mat2
R22_mat2 = Lambda2 + 2 * Mu2
R23_mat2 = Lambda2
R24_mat2 = 0
R25_mat2 = 0
R26_mat2 = 0

R31_mat2 = R13_mat2
R32_mat2 = R23_mat2
R33_mat2 = Lambda2 + 2 * Mu2
R34_mat2 = 0
R35_mat2 = 0
R36_mat2 = 0

R41_mat2 = 0
R42_mat2 = 0
R43_mat2 = 0
R44_mat2 = Mu2
R45_mat2 = 0
R46_mat2 = 0

R51_mat2 = 0
R52_mat2 = 0
R53_mat2 = 0
R54_mat2 = 0
R55_mat2 = Mu2
R56_mat2 = 0

R61_mat2 = 0
R62_mat2 = 0
R63_mat2 = 0
R64_mat2 = 0
R65_mat2 = 0
R66_mat2 = Mu2

R_mat2 = np.array([
    [R11_mat2, R12_mat2, R13_mat2, R14_mat2, R15_mat2, R16_mat2],
    [R21_mat2, R22_mat2, R23_mat2, R24_mat2, R25_mat2, R26_mat2],
    [R31_mat2, R32_mat2, R33_mat2, R34_mat2, R35_mat2, R36_mat2],
    [R41_mat2, R42_mat2, R43_mat2, R44_mat2, R45_mat2, R46_mat2],
    [R51_mat2, R52_mat2, R53_mat2, R54_mat2, R55_mat2, R56_mat2],
    [R61_mat2, R62_mat2, R63_mat2, R64_mat2, R65_mat2, R66_mat2],
])
