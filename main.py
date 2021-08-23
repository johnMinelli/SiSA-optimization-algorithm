import math
import time
from copy import deepcopy

import numpy as np

from rendering import render_search_results
from prof_clean import SIMPLEX

#  **standard values**
#  https://www.researchgate.net/publication/250772567_DOWNHILL_SIMPLEX_METHODS_FOR_OPTIMIZING_SIMULATED_ANNEALING_ARE_EFFECTIVE
#  **remarks on optimality**
#  https://www.hindawi.com/journals/mpe/2018/6193649/
#  It has been remarked [1, 5] that no feasible solution to the CPP exists with less than [m/t] grids, and
#  there is no optimal solution to the CPP with more than m grids. Hence, define sigma as the set of m-by-k
#  sub matrices of A without zero rows, with, satisfying [m/t] <= k <= m.
#  Now, for B belonging to sigma, denote by X'(B) the optimal solution of (Q) problem, and denote by X(B)
#  the rounded up, optimal solution of problem (Q) when solved without integrality constraints (7).
#  From the previous paragraph not only the pair (B,X(B)) is a feasible solution to the CPP, but
#  there exists B* belonging to sigma such that (B*, X'(B*)) is an optimal solution to the CPP.


class SISA_Optimizer:
    def __init__(self, number_covers, demands, grid_size, cost_covers, cost_grid):
        # standard values from literature then tuned  --> **standard values**
        self.INIT_TEMP = 10000
        self.MIN_TEMP = 100
        self.EXPLORE_ITERATIONS = 500
        self.THERMO_DEPENDANCE = 1
        self.COOLING_FACTOR = .8
        self.unfeasible = 0

        self.number_covers = number_covers
        self.demands = demands
        self.grid_size = grid_size
        self.cost_covers = cost_covers
        self.cost_grid = cost_grid
        self.linprog_algo = SIMPLEX()

    def optimize(self, debug=False, render=False):
        self.unfeasible = 0
        S = S_iter = self.initial_solution()
        cost_iter = cost_best = self.INIT_TEMP = self.evaluate_cost(S)
        self.MIN_TEMP = self.INIT_TEMP*0.01
        temperature = self.INIT_TEMP
        optimal_solutions = []

        S_best = self.minimize(S_iter)
        while (temperature >= self.MIN_TEMP):
            n_iter = 0
            randomness = 0
            cost_improving = cost_best
            while n_iter < self.EXPLORE_ITERATIONS:
                try:
                    covers = self.create_covers_sol(len(S_iter[1])) if n_iter == 0 \
                        else self.neighbourhood(S_iter, cost_iter, cost_best)
                    S = (covers, self.simplex(covers))
                    if debug: print("Solution for\n", S[0], "\n", S[1])
                except Exception as e:
                    self.unfeasible += 1
                    continue
                cost_current = self.evaluate_cost(S)
                if debug: print(f"$: {cost_current} / {cost_best}⍟ , {int(temperature)}°")
                if cost_current < cost_iter or (np.random.random() < np.exp(-((cost_current+1-cost_best)/cost_best)/(self.THERMO_DEPENDANCE*(temperature/self.INIT_TEMP)))):
                    if cost_current < cost_iter: randomness+=1
                    S_iter = S
                    cost_iter = cost_current
                    if cost_iter < cost_best:
                        S_best = self.minimize(S_iter)
                        cost_best = cost_iter
                        optimal_solutions.append((len(S_best[1]), cost_best, temperature))
                else: S_iter = self.resize(S_iter, cost_current, cost_iter, temperature)
                n_iter += 1
            temperature = self.update_temperature(temperature, (cost_improving+1-cost_best) / cost_best)
            if debug: print("Random explorations:", randomness)
            if debug: print(f"\n ANNEALING --> {int(temperature)}°")
        if debug: print("--"*40, "\nTO REFINE:\n", S_best[0], "\nCOST:", S_best[1],)
        refined_solution, cost = self.refine(S_best, debug)
        optimal_solutions.append((len(refined_solution[1]), cost, 0))
        
        if debug: print("OPT:\n", refined_solution, "\nCOST:", cost)
        if self.unfeasible != 0: print(f"Found {self.unfeasible}  unfeasible linear problem/s")
        if render: render_search_results(optimal_solutions)
        return cost

    def evaluate_cost(self, solution):
        covers_printed, grids_printed = solution
        # amount_covers_printed = np.sum(grids_printed * covers_printed, axis=1)
        cost_c1 = np.sum(grids_printed) * self.cost_covers if not(math.inf in grids_printed) else math.inf
        cost_c2 = self.cost_grid * np.count_nonzero(grids_printed) if not(math.inf in grids_printed) else math.inf
        tot = cost_c1+cost_c2
        return tot

    def initial_solution(self):
        '''
        :return: a cover displacement a matrix and the n° of prints for each grid
        '''
        while 1:
            try:
                # expect a mxm matrix
                covers_printed = self.create_covers_sol(self.number_covers)
                # expect m length array 
                grids_printed = self.simplex(covers_printed)
                # return solution tuple
                return (covers_printed, grids_printed)
            except Exception as e:
                self.unfeasible += 1
                continue

    def neighbourhood(self, actual_solution, cost_actual, cost_best):
        '''

        :param actual_solution: the current solution in the iteration cycle (covers displacement, n° print for grids)
        :param cost_actual: the cost of the current solution
        :param cost_best: the cost of the best solution found
        :return: a modified cover displacement matrix

        given a solution (B, X(B)) the tilde (modified) version is obtained by:
        - remove a grid by a P factor (the more optimal the solution is the more the possibility)
        - apply one of the two possible perturbations
        (1) in a column one is decremented and another is incremented
        (2) in a column one is decremented and the same is incremented in another column, moreover another
              another is incremented in the first column and the same is decremented in the second column
        '''
        covers_modified, grids  = deepcopy(actual_solution)
        k = len(grids)
        # reduce space
        cost_norm = (cost_actual - cost_best) / cost_best
        if np.min(grids)==0 and (k > (self.number_covers/self.grid_size)):
            if np.random.random() < np.exp(-cost_norm):  # probability based
                covers_modified = np.delete(covers_modified, np.argmin(grids), 1)
                covers_modified = self.increment_zero_rows(covers_modified)

        explore_neighborhood = True
        initial_matrix = covers_modified
        while explore_neighborhood or not(np.all(covers_modified>=0)):
            explore_neighborhood = False
            covers_modified = initial_matrix.copy()
            # perturbation
            possible_x = [*range(len(covers_modified))]
            possible_y = [*range(len(covers_modified[0]))]
            x1 = possible_x.pop(np.random.randint(0, len(possible_x)))
            x2 = possible_x.pop(np.random.randint(0, len(possible_x)))
            y1 = possible_y.pop(np.random.randint(0, len(possible_y)))
            y2 = possible_y.pop(np.random.randint(0, len(possible_y))) if len(possible_y) > 0 else None
            if np.random.random() > 0.5 and y2:   # single perturbation
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
        covers, grids = deepcopy(best_solution)
        indexes = np.argwhere(np.array(grids) == 0).flatten()
        for i in reversed(indexes):
            covers = np.delete(covers, i, 1)
            grids = np.delete(grids, i)
        return covers, grids

    def resize(self, actual_solution, cost_current, cost_actual, temperature):
        '''
        :param actual_solution: the solution in the iteration cycle (covers displacement, n° print for grids)
        :param cost_current: the cost of the discarded bad solution which caused the call 
        :param cost_actual: the cost of the solution in the iteration cycle
        :param temperature: temperature of optimization process
        :return: a resized version of the actual solution (B, X(B)) --> (B', X(B'))

        The function is called when the solution is bad and there is no luck in the solution acceptance.
        Modify the cover displace in a way to explore a more wide neighbourhood adding a column with random values
        (the matrix cover need to remain feasible)
        the resize happen if the all the grids are printed or by a factor of probability:
        - the more the n* of grids the less the probability of add more
        - the lower the difference cost with optimal solution the less the prob of enlarge the search space (matrix)
        - few grids and a bad (high) cost imply an high probability of enlarge the matrix
        - the temperature is an inverse multiplicative factor (decreasing augment the probability)
        --> see the docs for a detailed table of probability
        '''

        covers, grids = actual_solution
        k = len(grids)
        q = len(np.argwhere(np.array(grids) == 0).flatten())
        cols_norm = k / self.number_covers
        cost_norm = (cost_current +1 - cost_actual) / cost_actual
        temp_norm = self.THERMO_DEPENDANCE * (temperature/self.INIT_TEMP)
        if k < self.number_covers: 
            if q == 0 or np.random.random() < np.exp(-(cols_norm/(cost_norm*2))-(temp_norm*2)):  # probability based
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
            k = width_solution - for optimality must respect [m/t] <= k <= m   --> **remarks on optimality**
            t = self.grid_size
            bij is the number of instances of a cover in a grid displacement (each column)
            matrix constraints: bij value in the set of [0..t]; no zero rows; all columns sums up to t
        '''
        while True:
            sol = np.random.randint(0, self.grid_size, (self.number_covers,width_solution))
            for col_index, column in enumerate(sol.T):
                sum = np.sum(column)
                while sum != self.grid_size:
                    row_index = np.random.randint(0, self.number_covers)
                    if sum > self.grid_size and column[row_index] != 0:
                        sol[row_index, col_index] -= 1
                    elif sum < self.grid_size and column[row_index] != self.grid_size:
                        sol[row_index, col_index] += 1
                    sum = np.sum(column)
            sol = self.increment_zero_rows(sol)
            return sol

    def simplex(self, covers):
        '''
        :param covers: matrix with displacement of covers
        :return: a list with the n° of prints for each grid

        The linprog_algo is called with the method with constraints of the problem and the objective function
        A = covers matrix, the constraints to satisfy
        b = the demands for each cover, the bounds for the constraints
        c = np.ones(len(covers[0])), the objective function
        and it returns X(B)=[x1..xk] where each xi represent the n° of prints of each grid (basic vars of the simplex)
        the vector is a feasible not rounded (not integer constrained) solution
        '''

        c = np.ones(len(covers[0]))
        A = covers
        b = self.demands
        grids_solution = self.linprog_algo.solve(c, A, b)
        return grids_solution

    def refine(self, best_solution, debug):
        '''
        :param best_solution: best solution found (covers displacement, n° prints of grids)
        :param debug: flag for printout
        :return: refined best solution
        An additional cycle on the best solution found, with 0 acceptance probability and exploration
        limited to small perturbations
        '''
        S_best = best_solution
        cost_best = self.evaluate_cost(S_best)
        i = 0
        while i < self.EXPLORE_ITERATIONS:
            try:
                covers = self.neighbourhood(S_best, cost_best, cost_best)
                S = (covers, self.simplex(covers))
                if debug: print("Solution for\n", S[0], "\n", S[1])
            except Exception as e:
                self.unfeasible += 1
                continue
            cost_current = self.evaluate_cost(S)
            if debug: print("$:", cost_current, "/", cost_best, "⍟")
            if cost_current < cost_best:
                S_best = self.minimize(S)
                cost_best = cost_current
            i+=1
        return S_best, cost_best

    def update_temperature(self, temperature,  cost_reduction_norm):
        '''
        :param temperature: current temperature of the search process
        :param cost_reduction_norm: cost improvement of the solution, in the cycle: (cost_start-cost_best) normalized
        :return:
        The temperature decrease is dependent on a constant factor and an the cost_improvement
        The cost_improvement value is weighted incrementally with the decrease of the temperature
        '''
        temp_ratio = temperature/self.INIT_TEMP
        cost_val = cost_reduction_norm*(1+abs(np.log(temp_ratio)))
        return (self.COOLING_FACTOR-cost_val) * temperature

    def increment_zero_rows(self, covers_grids):
        while not np.all(np.sum(covers_grids, axis=1)):
            if max(np.sum(covers_grids, axis=1)) == 1:
                print("Error detection", covers_grids)
                covers_grids = self.create_covers_sol(len(covers_grids[0]))
            else:
                row_to_decrement = np.argmax(np.sum(covers_grids, axis=1))
                row_to_increment = np.argmin(np.sum(covers_grids, axis=1))
                col = np.argmax(covers_grids[row_to_decrement])
                covers_grids[row_to_decrement, col] -= 1
                covers_grids[row_to_increment, col] += 1
        return covers_grids


if __name__ == '__main__':
    N_INSTANCE = "001"

    with open(f"in/I{N_INSTANCE}.in", "r") as f:
        number_covers = int(f.readline())
        grid_size = int(f.readline())
        demands = []
        for a in range(number_covers):
            demands.append(int(f.readline()))
        c1, c2 = [float(i) for i in f.readline().split(" ")]
    start = time.time()
    sisa_cost = SISA_Optimizer(number_covers, demands, grid_size, c1, c2).optimize(debug=True, render=True)
    stop = time.time()

    with open(f"out/I{N_INSTANCE}.out", "r") as f:
        lines = f.read().splitlines()
        cost = float(lines[-1])

    print("Solved in {} s, at {}%".format(round(stop-start, 2), round(100-((sisa_cost-cost)*100/cost), 2)))

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



