import numpy as np

class task:
    def __init__(self, observation_space, timesteps = 0, balance_start = 0):
        self.action_space = 2  #Количество выходов
        self.timesteps = timesteps #Окно с данными для LSTM
        self.observation_space = observation_space #Количество входов
        self.balance_start = balance_start #Стартовый баланс
        self.balance = self.balance_start  #Баланс
        self.state = None #Текущее состояние
        self.log = [] #Логи по задаче

    def step(self, num = 0, action):
        """
        Расчитывает следующий шаг игры, вознаграждение за совершённое действие
        и признак завершения текущей игры.
        :param int num: Номер текущего примера для обучения.
        :param int action: Действие.
        :return: tuple(ndarray, float, bool, dict)
        """
        #Расчёт следующего шага

        #Расчёт вознаграждения
        reward = 0

        done = self.game_over()
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
        return np.array(self.state)

    def game_over(self):
        """
        Условие на завершение текущей игры.
        :return: bool
        """
        done = True if self.balance <= 0 else False
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
