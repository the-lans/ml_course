import numpy as np
import math

ENV_NAME = "CartPole" #Название задачи
FILE_NAME_LOG = "log.csv" #Файл с логами

class CartPole:
    """
        Description:
            A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

        Source:
            This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

        Observation:
            Type: Box(4)
            Num	Observation                 Min         Max
            0	Cart Position             -4.8            4.8
            1	Cart Velocity             -Inf            Inf
            2	Pole Angle                 -24 deg        24 deg
            3	Pole Velocity At Tip      -Inf            Inf

        Actions:
            Type: Discrete(2)
            Num	Action
            0	Push cart to the left
            1	Push cart to the right

            Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

        Reward:
            Reward is 1 for every step taken, including the termination step

        Starting State:
            All observations are assigned a uniform random value in [-0.05..0.05]

        Episode Termination:
            Pole Angle is more than 12 degrees
            Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
            Episode length is greater than 200
            Solved Requirements
            Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
        """

    def __init__(self, balance_start = 0):
        self.action_space = 2  #Количество выходов
        self.timesteps = 0 #Окно с данными для LSTM
        self.observation_space = 4 #Количество входов
        self.balance_start = balance_start #Стартовый баланс
        self.balance = self.balance_start  #Баланс
        self.state = None #Текущее состояние
        self.log = [] #Логи по задаче

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.steps_beyond_done = None

    def step(self, action, num = 0):
        """
        Расчитывает следующий шаг игры, вознаграждение за совершённое действие
        и признак завершения текущей игры.
        :param int num: Номер текущего примера для обучения.
        :param int action: Действие.
        :return: tuple(ndarray, float, bool, dict)
        """
        #Расчёт следующего шага
        (x, x_dot, theta, theta_dot) = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)

        #Расчёт вознаграждения
        done = self.game_over(x, theta)
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0

        self.balance += reward
        if done:
            self.log.append({"Balance": self.balance, "Reward": reward, "Num": num})
        return (np.array(self.state), reward, done, {})

    def reset(self, num = 0):
        """
        Сбрасывает игру в текущее состояние.
        :param int num: Номер текущего примера для обучения.
        :return: tuple(ndarray, float, bool, dict)
        """
        self.balance = self.balance_start
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(self.observation_space,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def game_over(self, x, theta):
        """
        Условие на завершение текущей игры.
        :return: bool
        """
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)
        return done

    def getBalance(self):
        """
        Текущий баланс игры.
        :return: float
        """
        return self.balance

    def getStartNum(self):
        """
        Номер примера, с которого осуществляется старт.
        :return: int
        """
        return self.timesteps
