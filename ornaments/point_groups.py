import numpy as np
from .utils import affine_to_projective


def t(x, y):
    return affine_to_projective(b=(x, y))


e = np.eye(3)

r2 = np.array([[-1, 0, 0],
               [0, -1, 0],
               [0, 0, 1]])

r3 = np.array([[0, -1, 0],
               [1, -1, 0],
               [0, 0, 1]])

r4 = np.array([[0, -1, 0],
               [1, 0, 0],
               [0, 0, 1]])

r6 = np.array([[1, -1, 0],
               [1, 0, 0],
               [0, 0, 1]])

f_p = np.array([[1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]])

f_c = np.array([[0, 1, 0],
                [1, 0, 0],
                [0, 0, 1]])

f_l = np.array([[1, 0, 0],
                [1, -1, 0],
                [0, 0, 1]])

f_s = np.array([[1, -1, 0],
                [0, -1, 0],
                [0, 0, 1]])

# rotate at the center
r2 = t(0.5, 0.5) @ r2 @ t(-0.5, -0.5)
r4 = t(0.5, 0.5) @ r4 @ t(-0.5, -0.5)

# mirror in the middle
f_p = t(0, 0.5) @ f_p @ t(0, -0.5)

# cyclic groups
C1 = [e]
C2 = [e, r2]
C3 = [e, r3, r3 @ r3]
C4 = [np.linalg.matrix_power(r4, n) for n in range(4)]
C6 = [np.linalg.matrix_power(r6, n) for n in range(6)]

# Dihedral groups
D1_p = C1 + [r @ f_p for r in C1]
D1_c = C1 + [r @ f_c for r in C1]
D2_p = C2 + [r @ f_p for r in C2]
D2_c = C2 + [r @ f_c for r in C2]
D3_l = C3 + [r @ f_l for r in C3]
D3_s = C3 + [r @ f_s for r in C3]
D4 = C4 + [r @ f_p for r in C4]
D6 = C6 + [r @ f_s for r in C6]


point_group_vectors = {
    "Pg": [(0, 0), (0.5, 0)],
    "Pmg": [(0, 0), (0, -0.5), (0, 0), (0, 0.5)],
    "Pgg": [(0, 0), (0, 0), (0.5, -0.5), (-0.5, 0.5)],
    "P4g": [(0, 0), (-0.5, 0.5), (0, 0), (0.5, -0.5), (0.5, -0.5), (0, 0), (-0.5, 0.5), (0, 0)],
}


point_groups = {
    "P1": C1,
    "P2": C2,
    "Pm": D1_p,
    "Pg": D1_p,
    "Cm": D1_c,
    "Pmm": D2_p,
    "Pmg": D2_p,
    "Pgg": D2_p,
    "Cmm": D2_c,
    "P4": C4,
    "P4m": D4,
    "P4g": D4,
    "P3": C3,
    "P3m1": D3_l,
    "P31m": D3_s,
    "P6": C6,
    "P6m": D6
}


for group, vectors in point_group_vectors.items():
    point_groups[group] = [t(tx, ty) @ g for (tx, ty), g in zip(vectors, point_groups[group])]
