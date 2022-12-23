import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import pyqtgraph.exporters
from random import randrange, uniform
import time
import neat


class Player:
    def __init__(self, x, y, fi=0, v=0, maxdfi=0, minv=0, maxv=0):
        self.x = x
        self.y = y
        self.fi = fi
        self.v = v
        self.maxdfi = maxdfi
        self.minv = minv
        self.maxv = maxv
        self.x_history = []
        self.y_history = []
        self.score = 0

    def init(self, x, y, fi=0, v=0, maxdfi=0, minv=0, maxv=0):
        self.__init__(x, y, fi=0, v=0, maxdfi=0, minv=0, maxv=0)

    def new_position(self):
        self.x += self.v * np.cos(self.fi)
        self.y += self.v * np.sin(self.fi)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        return self.x, self.y

    def direction_to(self, player):
        dx, dy = player.x - self.x, player.y - self.y
        return np.sign(dy) * np.arccos(dx / np.sqrt(dx * dx + dy * dy))

    def distance_to(self, player):
        dx, dy = player.x - self.x, player.y - self.y
        return np.sqrt(dx * dx + dy * dy)


class Prey(Player):
    def __init__(self, x, y, fi=0, v=0, maxdfi=0, minv=0, maxv=0, net=None):
        super().__init__(x, y, fi, v, maxdfi, minv, maxv)
        self.net = net

    def set_ai(self, net):
        self.net = net

    def decision(self, predator, target):
        self.fi += self.maxdfi * self.net.activate((self.direction_to(predator), self.distance_to(predator),
                                                    self.direction_to(target), self.distance_to(target)))[0]
        # dfi = self.direction_to(target) - self.fi
        # self.fi += np.sign(dfi) * min(self.maxdfi, abs(dfi))
        # return self.fi


class Predator(Player):
    def decision(self, prey):
        dfi = self.direction_to(prey) - self.fi
        if abs(dfi) > np.pi:
            dfi -= np.sign(dfi) * 2 * np.pi
        self.fi += np.sign(dfi) * min(self.maxdfi, abs(dfi))
        return self.fi


class Target(Player):
    pass


class Game:
    predator_dfi_max = 16
    predator_dfi_min = 15
    predator_v_max = 1.31
    predator_v_min = 1.3
    predator_start_max = 50.1
    predator_start_min = 50.0
    prey_start_x = 20
    prey_start_y = 20
    prey_dfi = 10
    prey_v = 1
    target_x = 80
    target_y = 80

    def __init__(self, net, n_steps=100, r_collision=0, Lx=100, Ly=100):
        self.prey = Prey(self.prey_start_x, self.prey_start_y, fi=np.pi/4,
                         maxdfi=np.pi / self.prey_dfi, v=self.prey_v, net=net)
        self.predator = Predator(uniform(self.predator_start_min, self.predator_start_max),
                                 0*uniform(self.predator_start_min, self.predator_start_max),
                           maxdfi=np.pi / randrange(self.predator_dfi_min, self.predator_dfi_max),
                           v=uniform(self.predator_v_min, self.predator_v_max))
        self.target = Target(self.target_x, self.target_y)
        self.n_steps = n_steps
        self.x_collisions = []
        self.y_collisions = []
        self.r_collision = r_collision
        self.n_collisions = 0
        self.Lx = Lx
        self.Ly = Ly

    def is_collision(self, prey, predator):
        if prey.distance_to(predator) <= self.r_collision:
            self.x_collisions.append(prey.x)
            self.y_collisions.append(prey.y)
            return True
        return False

    def is_outside(self, player):
        return abs(player.x) > self.Lx or abs(player.y) > self.Ly

    def calculate_score(self, prey, predator, target):
        prey.score += 1
        if self.is_collision(prey, predator):
            prey.score -= 10000
        if self.is_outside(prey):
            prey.score -= 1000
        if self.is_collision(prey, target):
            prey.score += 10000
        return prey.score

    def run(self, n_steps=None):
        if n_steps is not None:
            self.n_steps = n_steps
        step = 0
        game_over = False
        while step < self.n_steps and not game_over:
            self.prey.decision(self.predator, self.target)
            self.predator.decision(self.prey)
            self.prey.new_position()
            self.predator.new_position()
            self.prey.score = self.calculate_score(self.prey, self.predator, self.target)
            if self.is_collision(self.prey, self.predator):
                game_over = True
                self.n_collisions += 1
            if self.is_collision(self.prey, self.target):
                game_over = True
            step += 1
        self.prey.score -= 150 * int(self.prey.distance_to(self.target))
        return self.prey.score


def run_evolution(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        game = Game(net, n_steps=300, r_collision=1)
        genome.fitness = game.run()


def show_plot(prey, predator, target, p1):
    # import sys
    # if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
    #     app = pg.QtWidgets.QApplication(sys.argv)

    # p1 = pg.plot()
    p1.plot([target.x], [target.y], symbol='o', symbolPen='g', symbolBrush=None, symbolSize=15,
            clear=False)
    for i in range(0, len(prey.x_history)):
        p1.plot([prey.x_history[i]], [prey.y_history[i]], symbol='o', symbolPen='g', symbolBrush='g', symbolSize=5,
                clear=False)
        p1.plot([predator.x_history[i]], [predator.y_history[i]], symbol='o', symbolPen='r', symbolBrush='r',
                symbolSize=5, clear=False)
        time.sleep(0.02)
        pg.QtWidgets.QApplication.processEvents()
    # app.exec_()


def main():
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    # init NEAT
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # run NEAT
    winner = p.run(run_evolution, 500)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    ts = np.arange(0, len(stats.get_fitness_median()), 1).tolist()
    fig, ax = plt.subplots()
    ax.plot(ts, stats.get_fitness_median())
    plt.show()

    for i in range(0, 1):
        game = Game(winner_net, n_steps=300, r_collision=1)
        score = game.run()
        print(f"{score=}, {game.n_collisions=}")
        p1 = pg.plot()
        show_plot(game.prey, game.predator, game.target, p1)




if __name__ == '__main__':
    import sys

    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        app = pg.QtWidgets.QApplication(sys.argv)
    main()
    app.exec_()
