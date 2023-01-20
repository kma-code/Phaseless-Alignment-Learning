from matplotlib import pyplot as plt
from matplotlib import animation
import seaborn as sns


def animate_matrix_change(matrices, file_path='animation.gif', fps=60):
    """
    Plot a list of equally shaped matrices as an animation.
    """

    assert len(matrices) >= 2

    fig = plt.figure()
    frames_qty = len(matrices) - 1

    def init():
        sns.heatmap(matrices[0], vmax=.8, square=True)

    def animate(i):
        plt.clf()
        sns.heatmap(matrices[i + 1], vmax=.8, square=True)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames_qty, repeat = False)
    anim.save(file_path, writer='pillow', fps=fps)


