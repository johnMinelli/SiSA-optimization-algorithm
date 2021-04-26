import math
import numpy as np
import matplotlib.pyplot as plt

from rendering import render_inequalities
from simplex import SIMPLEX

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

def RAWsimplex(B):
    '''
    
    :param B: initial sol is (m by m); during execution is (m by k), m/t<=k<=m
    :return: 
    '''
    # call method with constraints of the problem
    # returns X(B)=[x1..xk] where each xi represent the number of prints of each grid (basic vars of the simplex)
    # the vector is a feasible not rounded (not integer constrained) solution
    # the solution S = (B,X(B)) and the z(S) cost is returned where
    # z(S) = sumij( bij * xij * Ci) + C2xn>0 where n>0 stands for the number of positive entries in X(B)
    # c = np.ones(B.size[1])
    # A = B
    # b = demand
    # solution rounded up
    
    # ** this is the structure **
    #     A = np.array([[1, 2, 3, 0],
    #                   [-1, 2, 6, 0],
    #                   [0, 4, 9, 0],
    #                   [0, 0, 3, 1]])
    #     b = np.array([1a, 1b, 1c, 1d])
    #     demand =  np.array([d1,d2,d3,d4])
    # 
    # [[1a1, 1b2, 1c3, 1d0],>= d1
    #  [1a-1, 1b2, 1c6, 1d0], >= d2
    #  [1a0, 1b4, 1c9, 1d0],>= d3
    #  [1a0, 1b0, 1c3, 1d1]])>= d4

def RAWneighbor(size):
    '''
    
    :param size: 
    :return: 
    
    # given a solution (B, X(B)) the tilde version is obtained by applying due possible actions
    # (1) in a column one is decremented and another is incremented
    # (2) in a column one is decremented and the same is incremented in another column, moreover another 
    #       another is incremented in the first column and the same is decremented in the second column  
    '''

def RAWresize(size):
    ''''''
    # (B, X(B)) --> (B', X(B'))
    # q = number of zeros in X(B)
    # if q > 1 remove the first q-1 columns =0 
    # if q = 1 ok 
    # if q = 0 and B has less than m columns --> add a column
    # (somma righe = t; differenza tra elementi in valore assoluto <= 1; se la domanda della riga è maggiore rispetto ad un'altra riga, allora anche il numero
    # di instanze della prima riga della nuova colonna sarà più alto rispetto a quello della seconda riga)
    # reexecute the simplex

def RAWinitial_feasible(size):
    '''
    # **B0**
    # mxm
    # bij in the set of [0..t]
    # no zero rows
    # all columns sums up to t
    # then
    # **S0**
    # simplex refine the matrix as a solution
    # cost of the solution
    '''
    # ** procedure initial_sol **

def RAWcreate_solution(size):
    '''
    :param size: 
    :return:  (mxk) grid where each column represent a grid, each row is an array of instances of each cover
                aij is the number of instances of a coevr in a grid displacement

                **remarks on optimality**
                It has been remarked [1, 5] that no feasible solution to the CPP exists with less than [m/t] grids, and
                there is no optimal solution to the CPP with more than m grids. Hence, define sigma as the set of m-by-k
                sub matrices of A without zero rows, with, satisfying [m/t] <= k <= m.
                Now, for B belonging to sigma, denote by X'(B) the optimal solution of (Q) problem, and denote by X(B)
                the rounded up, optimal solution of problem (Q) when solved without integrality constraints (7).
                From the previous paragraph not only the pair (B,X(B)) is a feasible solution to the CPP, but
                there exists B* belonging to sigma such that (B*, X'(B*)) is an optimal solution to the CPP.
    '''

    def RAWoptimize(self):
        # The standard choices for these constants are A = C(S), B is 100 times smaller than the best lower bound on cost, I = 1000, D = 1 and E = 0.9.
        # S = candidate solution
        # C(x) = cost function
        '''
        #** raw version **
        S = initial_feasible()
        best = evaluate_cost(S)
        temperature = 100000
        Sact = resize()
        Sbest = Sact
        while (temperature >= min_temperature): [T > B]
            n_iter = 0
            while (n_iter < max_iter):[i < I]
                S = simplex(neighbor(Sact))
                if (evaluate_cost(S) < evaluate_cost(Sact) || [random < e^( (C(S')-C(S)) / (D*T))] ):
                    Sact = resize(S)
                n_iter += 1
                if (evaluate_cost(Sact) < evaluate_cost(Sbest)):
                    Sbest = Sact
                    n_iter = 0
            temperature = update_temperature() [T = E*T]
        '''


class SISA:
    def __init__(self, number_covers, demands, grid_size, cost_covers, cost_grid):
        # standard values from literature
        self.INIT_TEMP = 10000
        self.MIN_TEMP = 100
        self.EXPLORE_ITERATIONS = 1000
        self.THERMO_DEPENDANCE = 1
        self.COOLING_FACTOR = .9
        self.useless = 0

        self.number_covers = number_covers
        self.demands = demands
        self.grid_size = grid_size
        self.cost_covers = cost_covers
        self.cost_grid = cost_grid
        self.linprog_algo = SIMPLEX()

    def optimize(self):
        S = S_act = self.initial_solution()
        cost_actual = cost_best = self.INIT_TEMP = self.evaluate_cost(S)
        self.MIN_TEMP = self.INIT_TEMP*0.01
        temperature = self.INIT_TEMP

        S_best = self.minimize(S_act)
        while (temperature >= self.MIN_TEMP):
            n_iter = 0
            k_improving = len(S_best[1])
            while (n_iter < self.EXPLORE_ITERATIONS):
                try:
                    covers = self.create_covers_sol(len(S_act[1])) if n_iter == 0 \
                        else self.neighbourhood(S_act, cost_actual, cost_best, temperature)  # TODO forse il RAWresize aveva ragione è meglio togliere tutti tranne 1 gli zeri (le griglie non usate dalla soluzione del simplesso)
                    # il mio ragionamento era nel neighbourhood tolgo una colonna alla volta con probabilità crescente se la soluzione è buona e la temperatura è ancora alta per ridurre poco a poco le colonne
                    # ma vedo che con un problema grande rimangono molti zeri
                    S = (covers, self.simplex(covers))
                except Exception as e:
                    # TODO accadono ancora i casi unfeasible per il simplesso? togli i weee
                    print("weeeeee")
                    self.useless += 1
                    continue
                cost_current = self.evaluate_cost(S)
                if cost_current < cost_actual or (np.random.random() < np.exp(-(1 + cost_current - cost_best)/(self.THERMO_DEPENDANCE*temperature))):
                    S_act = S
                    cost_actual = cost_current
                    if(cost_actual < cost_best):
                        S_best = self.minimize(S_act)
                        if len(S_best[0][0])==2:
                            print("")
                        cost_best = cost_actual
                else: S_act = self.resize(S_act, cost_current, cost_actual, temperature) # TODO valuta enlargement probability mi sembra che si espanda troppo 
                n_iter += 1
            temperature = self.update_temperature(temperature, max(0, k_improving-len(S_best[1])))
        print(self.refine(S_best))

    def evaluate_cost(self, solution):
        covers_printed = solution[0]
        grids_printed = solution[1]
        cost_c1 = np.sum(np.sum(grids_printed * covers_printed, axis=1) * self.cost_covers) if not(math.inf in grids_printed) else math.inf
        cost_c2 = self.cost_grid * np.count_nonzero(grids_printed) if not(math.inf in grids_printed) else math.inf
        tot = cost_c1+cost_c2
        print("Cost: ", tot, " (", cost_c1, " , ", cost_c2,")")
        return tot

    def initial_solution(self):
        while 1:
            try:
                # expect a mxm matrix
                covers_printed = self.create_covers_sol(self.number_covers)
                # expect m length array 
                grids_printed = self.simplex(covers_printed)
                # return solution tuple
                return (covers_printed, grids_printed)
            except Exception as e:
                self.useless += 1
                print(self.useless)
                print("weeeeee")
                continue

    def neighbourhood(self, actual_solution, cost_actual, cost_best, temperature):
        covers_modified, grids  = actual_solution
        k = len(grids)
        # reduce space
        if np.min(grids)==0 and (k > (self.number_covers/self.grid_size)):
            if np.random.random() < np.exp(-(cost_actual - cost_best)/(self.THERMO_DEPENDANCE*temperature)):
                covers_modified = np.delete(covers_modified, np.argmin(grids), 1)
                covers_modified = self.increment_zero_rows(covers_modified)

        explore_neigh = True
        initial_matrix = covers_modified
        while explore_neigh or not(np.all(covers_modified>=0)):
            explore_neigh = False
            covers_modified = initial_matrix.copy()
            # perturbation
            possible_x = [*range(len(covers_modified))]
            possible_y = [*range(len(covers_modified[0]))]
            x1 = possible_x.pop(np.random.randint(0, len(possible_x)))
            x2 = possible_x.pop(np.random.randint(0, len(possible_x)))
            y1 = possible_y.pop(np.random.randint(0, len(possible_y)))
            y2 = possible_y.pop(np.random.randint(0, len(possible_y))) if len(possible_y)>0 else None
            if np.random.random()>0.5 and y2:   # single perturbation
                covers_modified[x1, y1] -= 1
                covers_modified[x1, y2] += 1
                covers_modified[x2, y1] += 1
                covers_modified[x2, y2] -= 1
            else:                               # cross perturbation
                covers_modified[x1, y1] -= 1
                covers_modified[x2, y1] += 1
        covers_modified = self.increment_zero_rows(covers_modified)
        return covers_modified

    def minimize(self, best_solution):
        covers, grids = best_solution
        indexes = np.argwhere(np.array(grids) == 0).flatten()
        for i in reversed(indexes):
            covers = np.delete(covers, i, 1)
            grids = np.delete(grids, i)
        return (covers, grids)

    def resize(self, actual_solution, cost_current, cost_actual, temperature):
        covers, grids = actual_solution
        k = len(grids)
        q = len(np.argwhere(grids == 0).flatten())
        if k < self.number_covers:
            if q == 0 or np.random.random() > np.exp(-(cost_current - cost_actual) / (self.THERMO_DEPENDANCE * temperature)):
                column = np.random.randint(0, self.grid_size, self.number_covers)
                sum = np.sum(column)
                while sum != self.grid_size:
                    if sum > self.grid_size:
                        column[np.argmax(column)] -= 1
                    elif sum < self.grid_size:
                        column[np.argmin(column)] += 1
                    sum = np.sum(column)
                ordered_demand = np.argsort(self.demands)
                column = column[ordered_demand]
                col_index = np.random.randint(0, k)
                covers = np.insert(covers, col_index, column, axis=1)
                grids.append(0)
        return covers, grids

    def create_covers_sol(self, width_solution):
        '''
        :param width_solution: number of columns of the return matrix
        :return: a matrix (m by k) where
            m = self.number_covers
            t = self.grid_size
            k = width_solution and for optimality must respect [m/t] <= k <= m
            bij is the number of instances of a cover in a grid displacement (each column)
            matrix constraints: bij value in the set of [0..t]; no zero rows; all columns sums up to t
        '''
        while True:
            sol = np.random.randint(0, self.grid_size, (self.number_covers,width_solution))
            for col_index, column in enumerate(sol.T):
                sum = np.sum(column)
                while sum != self.grid_size:
                    row_index = np.random.randint(0, self.number_covers)-1
                    if sum > self.grid_size and column[row_index] != 0:
                        sol[row_index, col_index] -= 1
                    elif sum < self.grid_size and column[row_index] != self.number_covers:
                        sol[row_index, col_index] += 1
                    sum = np.sum(column)
                sol = self.increment_zero_rows(sol)
                print("B0")
                print(sol)
                print(" -- sum columns: ", np.sum(sol, axis=0), " -- no zero rows: ", np.all(np.sum(sol, axis=1)))
                return sol

    def simplex(self, covers_solution):
        c = np.ones(len(covers_solution[0]))
        A = covers_solution
        b = self.demands
        grids_solution = self.linprog_algo.solve(c, A, b)
        print("Solution for\n", covers_solution, "\n", grids_solution)
        return grids_solution

    def refine(self, best_solution):
        # TODO verify this
        # TODO rounding to integer in simplex
        S_best = best_solution
        cost_best = self.evaluate_cost(S_best)
        i = 0
        while i < self.EXPLORE_ITERATIONS:
            try:
                covers = self.neighbourhood(S_best, cost_best, cost_best, 1)
                S = (covers, self.simplex(covers))
            except Exception as e:
                self.useless+=1
                print("weeeeee")
                continue
            cost_current = self.evaluate_cost(S)
            if cost_current < cost_best:
                S_best = self.minimize(S)
                cost_best = cost_current
            i+=1
        return (S_best, cost_best)

    def update_temperature(self, temperature, grids_improvement):
        grids_reduction = grids_improvement/(self.number_covers-(self.number_covers/self.grid_size))
        return (self.COOLING_FACTOR-grids_reduction) * temperature

    def increment_zero_rows(self, covers_grids):
        while not np.all(np.sum(covers_grids, axis=1)):
            if max(np.sum(covers_grids, axis=1)) == 1:
                # TODO remove that
                print(covers_grids)
                raise Exception("THIS IS NEEDED")
                covers_grids = self.create_covers_sol(len(covers_grids[0]))
            else:
                row_to_decrement = np.argmax(np.sum(covers_grids, axis=1))
                row_to_increment = np.argmin(np.sum(covers_grids, axis=1))
                col = np.argmax(covers_grids[row_to_decrement])
                covers_grids[row_to_decrement, col] -= 1
                covers_grids[row_to_increment, col] += 1
        return covers_grids


if __name__ == '__main__':
    with open("I005.in", "r") as f:
        number_covers = int(f.readline())
        grid_size = int(f.readline())
        demands = []
        for a in range(number_covers):
            demands.append(int(f.readline()))
        c1, c2 = [float(i) for i in f.readline().split(" ")]
        
    SISA(number_covers, demands, grid_size, c1, c2).optimize()
    
    # Toy example
    # m = 5 # n of covers
    # t = 16 # grid size (4 by 4)
    # demand = [12, 18, 44, 47]
    # prints = np.array([[1, 2, 3, 0],
    #                    [1, 2, 2, 7],
    #                    [6, 4, 0, 0],
    #                    [0, 0, 3, 1]])
    # printed = np.array([[1, 2, 3, 0],
    #                    [0, 0, 0, 0],
    #                    [0, 0, 0, 0],
    #                    [0, 0, 0, 0]])
    # print(demand * printed)
    # k = 3 # solution n of grids
    
    

