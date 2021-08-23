import math

import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, debug=False):
        self.debug = debug
        self.TOL = 1.0E-12

        # c = [1, 1, 1, 0, 0, 0, 0, 0]
        # A = [[2, 1, 0, -1, 0, 0, 0, 0],
        #      [0, 2, 0, 0, -1, 0, 0, 0],
        #      [2, 0, 1, 0, 0, -1, 0, 0],
        #      [0, 1, 1, 0, 0, 0, -1, 0],
        #      [0, 0, 2, 0, 0, 0, 0, -1]]
        # b = [84500, 94500, 103500, 114500, 135500]
        c = [1, 1]
        # A = [[0, 1],
        #      [4, 0],
        #      [0, 3]]
        # A = [[0, 1],
        #      [0, 2],
        #      [4, 1]]
        A = [[0, 2],
             [3, 1],
             [1, 1]]
        # A = [[3, 1, 0],
        #      [0, 2, 0],
        #      [1, 1, 4]]
        b = [4500, 9000, 16000]
        
        # example from geekrodion
        # /*A=[[-1, 1, 1, 0], [ 1, 0, 0, 1]]
        # b = [0, 2]
        # c = [0, 1, 0, 0]*/
        
        # pre normalization it raised an unbounded
        # A=[[0,1,0,1,0,0],
        #     [1,0,0,0,0,0],
        #     [0,0,0,0,1,0],
        #     [0,0,0,0,1,0],
        #     [0,1,0,0,0,1],
        #     [1,0,0,0,0,0],
        #     [0,0,0,0,0,1],
        #     [0,0,1,1,0,1],
        #     [0,1,1,1,1,0],
        #     [1,0,1,0,0,0],
        #     [0,1,1,1,0,0],
        #     [1,0,0,0,1,1]]
        # b = [1000, 1500, 2000, 2000, 2500, 3000, 4500, 5200, 6000, 7000, 9000, 12000]
        # c = [1.,1.,1.,1.,1.,1.]

        A=[[0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0],
            [1,0,0,0,1,0,0],
            [0,0,0,0,1,0,0],
            [0,0,1,0,0,0,0],
            [1,0,0,0,0,0,1],
            [0,0,0,1,0,1,1],
            [0,0,1,0,0,1,0],
            [0,1,1,0,0,0,1],
            [2,1,0,1,0,0,0],
            [0,1,0,1,1,1,0],
            [0,1,0,1,0,1,1]]
        b = [1000, 1500, 2000, 2000, 2500, 3000, 4500, 5200, 6000, 7000, 9000, 12000]
        c = [1.,1.,1.,1.,1.,1.,1.]


        try:
            solution = self.solve(c, A, b, render=True)
            print('Optimal solution: ', solution)
        except Exception as e:
            print(e)

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
        a_vars = np.arange(len(A[0]) - len(av[0]), len(A[0]))
        # artificial variables are also the basic variables
        basics = a_vars.copy()

        # the problem objective is a minimization. Turn it in max like that:  min Z=a1+a2 -> max Z=-a1-a2 -> Z+a1+a2=0
        # rearrange changing sign Z + x_1 + x_2 = 0  ; then complete with 0s
        c = np.hstack((c, [0] * len(surplus[0]), [0] * len(av[0])))

        return c, A, b, a_vars, basics

    def to_tableau_first_phase(self, c, A, b, a_vars, basics):
        xb = np.column_stack((A, b))
        # original objective function
        z = np.hstack((c, 0))
        # new obj function is a min of a sum of artificial var. Turn into max:  min Z=sum ai -> max Z=sum -ai -> Z+ai=0
        z1 = np.zeros(len(z))
        z1[a_vars] = 1  # transformed into a maximization
        # then expressed in form of non basic variables
        # sum of constraints where there is an a_var and subtract to make  'np.any(tab[-1, basics]==False'
        z1 = z1 - np.sum(xb[np.any(xb[:, a_vars] > 0, axis=1)], axis=0)
        tab = np.vstack((xb, z, z1))
        return np.array(tab, dtype='float')

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
        return any(x < -self.TOL for x in z[:-1])

    def get_pivot_position(self, tableau, basics):
        if self.debug: [print(' \t'.join(map(str, i))) for i in tableau]
        z = tableau[-1]
        column_index = np.argmin(z[:-1])
        # column_index = next(i for i, x in enumerate(z[:-1]) if x > self.TOL)

        restrictions = []
        for equation in tableau[:len(basics)]:
            factor = equation[column_index]
            restrictions.append(math.inf if factor <= 0 else equation[-1] / factor)

        min_ratio = min(restrictions)
        if min_ratio == math.inf:
            raise Exception("Unbounded")
        row_index = restrictions.index(min_ratio)
        if self.debug: print("Pivot row", row_index, " col", column_index)
        return row_index, column_index

    def pivot_step(self, tableau, pivot_position):
        i, j = pivot_position
        pivot_value = tableau[i, j]
        tableau[i] = [e / pivot_value for e in tableau[i]]

        for eq_i in range(len(tableau)):
            if eq_i != i:
                m = tableau[eq_i, j]
                multiplier = [e * m for e in tableau[i]]
                tableau[eq_i] = [e1 - e2 for e1, e2 in zip(tableau[eq_i], multiplier)]
        norm = np.array([np.array([v if abs(round(v,1)-v)>self.TOL else round(v,1) for v in r]) for r in tableau])
        if self.debug and not np.array_equal(tableau, norm):
            print("HERE a DIFFERENCE")
            print(tableau)
        return norm

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
        return -tableau[-1][-1]  # the change of sign is just to display better the solution

    def solve(self, c, A, b, print_result=False, render=False):
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
                pivot_position = self.get_pivot_position(tableau, basic_variables)
                tableau = self.pivot_step(tableau, pivot_position)
                basic_variables[pivot_position[0]] = pivot_position[1]
                zs.append(self.get_objective_function_value(tableau))
                solutions.append(self.get_solution(tableau, basic_variables))
                if phase1:
                    if any([opt<0 for opt in np.array(solutions[-1])[art_variables]]):
                        print("capture infeasible")  # TODO prova ad implementarlo (guarda geekrodion)

            if phase1:
                phase1 = False
                if any([art in basic_variables for art in art_variables]):
                    raise Exception("mmmh")
                if np.any(np.array(solutions[-1])[art_variables]):
                    raise Exception("Infeasible")
                if self.debug:
                    [print(' \t'.join(map(str, i))) for i in tableau]
                    print("--------------- End Phase 1 ---------------")
            elif phase2:
                phase2 = False

        if self.debug: print(tableau)
        if self.debug: print("Solution values: ", solutions[-1])
        if self.debug: print("Basic variables: ", basic_variables)
        if print_result: print("Solution: ", solutions[-1][:real_variables])

        if render:
            # for degeneracy detection
            steps = range(len(zs))
            plt.plot(steps, zs, color="#2c3e50")
            plt.xticks(steps)
            plt.xlabel('Steps')
            plt.ylabel('Objective function value')
            plt.title('Simplex')
            plt.show()

        return solutions[-1][:real_variables]


if __name__ == '__main__':
    SIMPLEX(debug=True)