import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, prey, predator, target, n_steps=100, r_collision=0, Lx=100, Ly=100):
        self.prey = prey
        self.predator = predator
        self.target = target
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
            self.n_collisions += 1
            return True
        return False

    def is_outside(self, player):
        return abs(player.x) > self.Lx or abs(player.y) > self.Ly

    def calculate_score(self, prey, predator, target):
        prey.score += 1
        if self.is_collision(prey, predator):
            prey.score -= 100
        if self.is_outside(prey):
            prey.score -= 10000
        if self.is_collision(prey, target):
            prey.score += 1000
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
            if self.is_collision(self.prey, self.target):
                game_over = True
            step += 1
        return self.prey.score


def run_evolution(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        prey = Prey(20, 20, maxdfi=np.pi / 10, v=1, net=net)
        target = Target(80, 80)
        predator = Predator(50, 0, maxdfi=np.pi / 20, v=1.2)
        game = Game(prey, predator, target, n_steps=150, r_collision=1)
        genome.fitness = game.run()

    fig, ax = plt.subplots()
    ax.plot(game.prey.x_history, game.prey.y_history, 'o', color='green')
    ax.plot(game.predator.x_history, game.predator.y_history, 'o', color='red')
    ax.plot(game.x_collisions, game.y_collisions, 'o', markersize=15, mfc='none', color='k')
    plt.show()


def main():
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)
    # init NEAT
    p = neat.Population(config)
    # run NEAT
    p.run(run_evolution, 10)


if __name__ == '__main__':
    main()
