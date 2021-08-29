import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def render_search_results(optimal_solutions):

    sols = np.array(optimal_solutions)
    t = sols[0][2]+1
    for sol in sols:
        if sol[2] >= t: sol[2] = t-10000
        t = sol[2]

    sizes = sols[:, 0]
    values = sols[:, 1]
    temps = sols[:, 2]
    color_map = 'coolwarm_r'
    cm = plt.get_cmap(color_map)
    n_sol = len(sizes)
    plt.close()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xticks(range(np.int(min(sizes)) if min(sizes)!=max(sizes) else np.int(min(sizes))-1, np.int(max(sizes))+1))
    ax.bar3d(sizes, temps, -np.zeros_like(sizes), 0.15, 25000, values, shade=True,
             color=[cm(1.0 * i / (n_sol - 1)) for i in range(n_sol)])

    for i in range(n_sol):
        ax.set_prop_cycle("c", [cm(1.0 * i / (n_sol - 1)) for i in range(n_sol - 1)])
        for i in range(n_sol - 1):
            plt.plot(sizes[i:i + 2], temps[i:i + 2], values[i:i + 2])
    ax.set_xlabel('N° grids')
    ax.set_ylabel('Temperature')
    ax.set_zlabel('Optimal cost')
    ax.set_title('SISA Search evaluation')
    plt.show()
    plt.close()