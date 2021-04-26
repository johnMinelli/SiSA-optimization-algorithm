from scipy.spatial import HalfspaceIntersection, ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

def render_inequalities(halfspaces, feasible_point, xlim, ylim):
    hs = HalfspaceIntersection(np.array(halfspaces), np.array(feasible_point))
    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x = np.linspace(*xlim, 100)

    for h in halfspaces:
        if h[1]== 0:
            ax.axvline(-h[2]/h[0], color="#2c3e50")
        else:
            ax.plot(x, (-h[2]-h[0]*x)/h[1], color="#2c3e50")
    x, y = zip(*hs.intersections)
    points = list(zip(x, y))
    convex_hull = ConvexHull(points)
    polygon = Polygon([points[v] for v in convex_hull.vertices], color="#34495e")
    ax.add_patch(polygon)
    ax.plot(x, y, 'o', color="#e67e22")

def printTableau(self, tableau):

    print("ind \t\t", end="")
    for j in range(0, len(c)):
        print("x_" + str(j), end="\t")
    for j in range(0, (len(tableau[0]) - len(c) - 2)):
        print("s_" + str(j), end="\t")

    print()
    for j in range(0, len(tableau)):
        for i in range(0, len(tableau[0])):
            if (not np.isnan(tableau[j, i])):
                if (i == 0):
                    print(int(tableau[j, i]), end="\t")
                else:
                    print(round(tableau[j, i], 2), end="\t")
            else:
                print(end="\t")
        print()

def getTableau(self):

    t1 = np.array([None, 0])
    numVar = len(self.c)
    numSlack = len(self.A)

    t1 = np.hstack(([None], [0], self.c, [0] * numSlack))

    basis = np.array([0] * numSlack)

    for i in range(0, len(basis)):
        basis[i] = numVar + i

    A = self.A

    if (not ((numSlack + numVar) == len(self.A[0]))):
        B = np.identity(numSlack)
        A = np.hstack((self.A, B))

    t2 = np.hstack((np.transpose([basis]), np.transpose([self.b]), A))

    tableau = np.vstack((t1, t2))

    tableau = np.array(tableau, dtype='float')
    
    return tableau