import math
from decimal import *
from fractions import Fraction

import numpy

import numpy as np
import matplotlib.pyplot as plt

from rendering import render_inequalities

'''
# We will consider only linear program in equational form such that
# 1. The system of equations Ax=b has at least one solution.
# 2. The rows of the matrix A are linearly independent.
      If some row of A is a linear combination of the other rows, then the corresponding equation is redundant and it can be deleted from the system without changing the set of equations.

Maximize cᵀx subject to Ax ≤ b and x≥0      (P) is equal to     Minimize bᵀy subject to Aᵀy ≥ c and y ≥ 0   (D)
    1. Nether (P) nor (D) has a feasible solution.
    2. (P) is unbounded and (D) has no feasible solution.
    3. (P) has no feasible solution and (D) is unbounded.
    4. Both (P) and (D) have a feasible solution. Then both have an optimal solution, and if x is an optimal solution of (P) and y is an optimal solution of (D), then cᵀx = bᵀy
    so, the maximum of (P) equals the minimum of (D).
'''


# model1 = LinearModel()
# 
# A = np.array([[1, 2, 3, 0],
#               [-1, 2, 6, 0],
#               [0, 4, 9, 0],
#               [0, 0, 3, 1]])
# b = np.array([3, 2, 5, 1])
# c = np.array([1, 1, 1, 0])
# 
# model1.addA(A)
# model1.addB(b)
# model1.addC(c)
# model1.setObj("MIN")
# model1.setPrintIter(True)
# 
# print("A =\n", A, "\n")
# print("b =\n", b, "\n")
# print("c =\n", c, "\n\n")
# model1.optimize()
# print("\n")
# model1.printSoln()

class SIMPLEX:
    # [[3 3 0], [1 0 2], [0 1 2]]
    def __init__(self):
        self.TOL = 1.0E-12

    #     # c = [1, 1, 1, 0, 0, 0, 0, 0]
    #     # A = [[2, 1, 0, -1, 0, 0, 0, 0],
    #     #      [0, 2, 0, 0, -1, 0, 0, 0],
    #     #      [2, 0, 1, 0, 0, -1, 0, 0],
    #     #      [0, 1, 1, 0, 0, 0, -1, 0],
    #     #      [0, 0, 2, 0, 0, 0, 0, -1]]
    #     # b = [84500, 94500, 103500, 114500, 135500]
    #     c = [1, 1]
    #     # A = [[0, 1],
    #     #      [4, 0],
    #     #      [0, 3]]
    #     # A = [[0, 1],
    #     #      [0, 2],
    #     #      [4, 1]]
    #     A = [[0, 2],
    #          [3, 1],
    #          [1, 1]]
    #     # A = [[3, 1, 0],
    #     #      [0, 2, 0],
    #     #      [1, 1, 4]]
    #     b = [4500, 9000, 16000]
    #     getcontext().prec = 4
    #     halfspaces = [
    #         [-1, 0, 0],  # x₁ ≥ 0
    #         [0, -1, 0],  # x₂ ≥ 0
    #         [-1, 1, -2],  # -x₁ + x₂ ≤ 2
    #         [1, 0, -4],  # x₁ ≤ 4
    #         [0, 1, -4],  # x₂ ≤ 4
    #     ]
    #     feasible_point = np.array([0.5, 0.5])
    # 
    #     try:
    #         solution = self.solve(c, A, b, halfspaces, feasible_point, show=True, debug=True)
    #         print('Optimal solution: ', solution)
    #     except Exception as e:
    #         print(e)

    def phase1_tableau(self, c, A, b):
        vars = len(c)
        a_vars = len(A)

        z = np.hstack(([0] * vars, [0] * a_vars, [0]))

    def to_standard_form(self, c, A, b):
        '''
        This function is dependant to the problem application
        :param c: problem objective function (minimization)
        :param A: problem constraints
        :param b: problem constants values of constraints
        :return: the standardized parameters and arrays of corresponding index for basic and artificial variables
        '''

        (n_cons, n_vars) = np.matrix(A).shape
        # All constants are already positive
        # All constraints are >= so add a surplus variable for each one
        surplus = -np.identity(n_cons, dtype=int)
        A = np.hstack((A, surplus))
        # moreover add an artificial variable for each constraint without slack variable
        av = np.identity(n_cons)
        A = np.hstack((A, av))
        a_vars = np.arange(len(A[0])-len(av[0]), len(A[0]))
        # artificial variables are also the basic variables
        basics = a_vars.copy()

        # the problem objective is a minimization so the obj function is turned into a maximization and completed with 0
        c = -np.hstack((c, [0] * len(surplus[0]), [0] * len(av[0])))
        
        return c, A, b, a_vars, basics

    def to_tableau_first_phase(self, c, A, b, a_vars, basics):
        xb = np.column_stack((A, b))
        # original objective function
        z = np.hstack((c, 0))
        # new objective function is a summation of artificial variables
        z1 = np.zeros(len(z))
        z1[a_vars] = -1  # transformed into a maximization
        # then expressed in form of non basic variables
        z1 = z1 + np.sum(xb[np.any(xb[:, a_vars] > 0, axis=1)], axis=0)  # sum of constraints where there is an a_var
        tab = np.vstack((xb, z, z1))
        if np.any(tab[-1, basics]):
            for eq_i in range(len(xb)):
                col_basic = basics[xb[eq_i, basics] > 0]
                multiplier = [e*tab[-1, col_basic] for e in tab[eq_i]]
                tab[-1] = [e1-e2 for e1,e2 in zip(tab[-1], multiplier)]
        return np.array(tab).astype(Fraction)

    def to_tableau_second_phase(self, tableau, a_vars):
        tab = np.delete(tableau, -1, 0)
        tab = np.delete(tab, a_vars, 1)
        return tab

    def can_be_improved_for_dual(self, tableau):
        rhs_entries = [row[-1] for row in tableau[:-1]]
        return any([entry < 0 for entry in rhs_entries])

    def get_pivot_position_for_dual(self, tableau):
        rhs_entries = [row[-1] for row in tableau[:-1]]
        min_rhs_value = min(rhs_entries)
        row = rhs_entries.index(min_rhs_value)

        columns = []
        for index, element in enumerate(tableau[row][:-1]):
            if element < 0:
                columns.append(index)
        columns_values = [tableau[row][c] / tableau[-1][c] for c in columns]
        column_min_index = columns_values.index(min(columns_values))
        column = columns[column_min_index]

        return row, column

    def can_be_improved(self, tableau):
        z = tableau[-1]
        return any(x > self.TOL for x in z[:-1])

    def get_pivot_position(self, tableau, basics, debug):
        if debug: print(tableau)
        z = tableau[-1]
        # column_index = np.argmax(z[:-1])
        column_index = next(i for i, x in enumerate(z[:-1]) if x > self.TOL)

        restrictions = []
        for equation in tableau[:len(basics)]:
            factor = equation[column_index]
            restrictions.append(math.inf if factor <= 0 else equation[-1] / factor)

        min_ratio = min(restrictions)
        if min_ratio == math.inf:
            raise Exception("Unbounded")
        row_index = restrictions.index(min_ratio)
        if debug: print("Pivot row", row_index, " col", column_index)
        return row_index, column_index

    def pivot_step(self, tableau, pivot_position):
        i, j = pivot_position
        pivot_value = tableau[i, j]
        tableau[i] = [e/pivot_value for e in tableau[i]]

        for eq_i in range(len(tableau)):
            if eq_i != i:
                m = tableau[eq_i, j]
                multiplier = [e*m for e in tableau[i]]
                tableau[eq_i] = [e1-e2 for e1, e2 in zip(tableau[eq_i], multiplier)]
        return tableau

    def is_basic(self, column):
        return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

    def get_solution(self, tableau, basics):
        columns = np.array(tableau).T
        solutions = []
        for col_index, column in enumerate(columns):
            solution = 0
            if self.is_basic(column) and col_index in basics:
                one_index = column.tolist().index(1)
                solution = round(columns[-1][one_index])
            solutions.append(solution)

        return solutions

    def get_objective_function_value(self, tableau):
        return -tableau[-1][-1]

    def solve(self, c, A, b, halfspaces=None, feasible_point=None, show=False, debug=False):
        real_variables = len(A[0])
        c, A, b, art_variables, basic_variables = self.to_standard_form(c, A, b)
        tableau = self.to_tableau_first_phase(c, A, b, art_variables, basic_variables)

        zs = [self.get_objective_function_value(tableau)]
        solutions = [self.get_solution(tableau, basic_variables)]

        phase1 = phase2 = True
        while phase1 or phase2:
            if phase2 and not phase1:
                tableau = self.to_tableau_second_phase(tableau, art_variables)
                solutions.append(self.get_solution(tableau, basic_variables))

            while self.can_be_improved(tableau):
                pivot_position = self.get_pivot_position(tableau, basic_variables, debug)
                tableau = self.pivot_step(tableau, pivot_position)
                basic_variables[pivot_position[0]] = pivot_position[1]
                zs.append(self.get_objective_function_value(tableau))
                solutions.append(self.get_solution(tableau, basic_variables))

            if phase1:
                phase1 = False
                if np.any(np.array(solutions[-1])[art_variables]):
                    raise Exception("Infeasible")
                if debug:
                    print("-------------------------------------")
            elif phase2:
                phase2 = False

        if debug: print(tableau)
        if debug: print("Solution values: ", solutions[-1])
        if debug: print("Basic variables: ", basic_variables)
        print("Solution: ", solutions[-1][:real_variables])

        if show:
            # for solutions rendering
            points = [v[:2] for v in solutions]
            xlim = (-1, max([p[0] for p in points]) + 1)
            ylim = (-1, max([p[1] for p in points]) + 1)
            render_inequalities(halfspaces, feasible_point, xlim, ylim)

            for start, end in zip(points[:-1], points[1:]):
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if dx > 0 or dy > 0:
                    plt.arrow(start[0], start[1], dx, dy, width=0.1, length_includes_head=True, color='#1abc9c')
            plt.show()

            # for degeneracy detection
            steps = range(len(zs))
            plt.plot(steps, zs, color="#2c3e50")
            plt.xticks(steps)
            plt.xlabel('steps')
            plt.ylabel('objective function value')
            plt.show()

        return solutions[-1][:real_variables]

if __name__ == '__main__':
    SIMPLEX()