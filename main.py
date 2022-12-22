import numpy as np
import matplotlib.pyplot as plt


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
    def decision(self, target):
        dfi = self.direction_to(target) - self.fi
        self.fi += np.sign(dfi) * min(self.maxdfi, abs(dfi))
        return self.fi


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
    def __init__(self, r_collision=0, Lx=100, Ly=100):
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


def main():
    game = Game(r_collision=1)
    prey = Prey(0, 0, maxdfi=np.pi / 10, v=1)
    target = Target(90, 90)
    predator = Predator(50, 0, maxdfi=np.pi / 20, v=1.1)
    maxt = 150
    t = 0
    while t < maxt:
        prey.decision(target)
        predator.decision(prey)
        prey.new_position()
        predator.new_position()
        print("score = ", game.calculate_score(prey, predator, target))
        if game.is_collision(prey, predator):
            print(f"Eat - {t}, {prey.x}, {prey.y}")
            break
        if game.is_collision(prey, target):
            print(f"Target - {t}, {prey.x}, {prey.y}")
            break
        t += 1

    fig, ax = plt.subplots()
    ax.plot(prey.x_history, prey.y_history, 'o', color='green')
    ax.plot(predator.x_history, predator.y_history, 'o', color='red')
    ax.plot(game.x_collisions, game.y_collisions, 'o', markersize=15, mfc='none', color='k')
    plt.show()


if __name__ == '__main__':
    main()
