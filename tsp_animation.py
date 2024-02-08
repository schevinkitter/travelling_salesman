import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animation_tsp(history, city_locations):

    fig, ax = plt.subplots()

    line, = plt.plot([], [], lw=2)

    def init():
        x, y = city_locations.T
        plt.plot(x, y, 'co')

        line.set_data([], [])

        return line,

    def update(frame):
        x = [city_locations[i, 0]
             for i in history[frame] + [history[frame][0]]]
        y = [city_locations[i, 1]
             for i in history[frame] + [history[frame][0]]]
        line.set_data(x, y)
        return line

    ani = FuncAnimation(fig, update, frames=range(0, len(history), len(history)//100),
                        init_func=init, interval=60, repeat=False)

    plt.show()

    return
