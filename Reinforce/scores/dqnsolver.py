import random
import numpy as np
from collections import deque
from scores.score_logger import ScoreLogger
from scores.model import *

GAMMA = 0.95 #Коэффициент функции вознаграждения
LEARNING_RATE = 0.001 #Скорость обучения
MEMORY_SIZE = 1000 #Размер памяти очереди примеров
RND_MEMORY = True #Выбирать примеры случайно из очереди?
BATCH_SIZE = 20 #Размер выборки
MAX_ITERATION = 100 #Максимальное количество игр
MAX_EXAMPLES = 10000 #Максимальное количество сформированных примеров
EXPLORATION_MAX = 1.0 #Первоначальный коэффициент эксплуатации
EXPLORATION_MIN = 0.01 #Минимальный коэффициент эксплуатации
EXPLORATION_DECAY = 0.995 #Уменьшение коэффициента эксплуатации
LOG_STATE = False #Вывод состояния

class DQNSolver:
    def __init__(self, env, env_name, max_iter, max_examples, lr, memsize = 1000000, bath_size = 32, gamma = 0.95,
                 exp_min = 0.01, exp_max = 1.0, exp_decay = 0.995, rnd_memory = True, log_state = False):
        self.modelnn = ModelNeuro(env, LAYERS, ACTIV, TYPE_NET, lr)
        self.env = env
        self.env_name = env_name
        self.max_iter = max_iter
        self.max_examples = max_examples
        self.log_state = log_state
        self.exploration_min = exp_min
        self.exploration_max = exp_max
        self.exploration_decay = exp_decay
        self.exploration_rate = self.exploration_max
        self.bath_size = bath_size
        self.gamma = gamma
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.rnd_memory = rnd_memory
        self.memory = deque(maxlen=memsize)
        self.num = 0
        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.exploration_rate = self.exploration_max
        self.num = self.env.getStartNum()
        self.memory.clear()

    def remember(self, state, action, reward, next_state, done):
        self.num += 1
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.modelnn.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.bath_size:
            return

        if self.rnd_memory:
            batch = random.sample(self.memory, self.bath_size)
        else:
            batch = list(self.memory)[-self.bath_size:]

        for (state, action, reward, state_next, terminal) in batch:
            q_update = reward if terminal else (reward + self.gamma * np.amax(self.modelnn.predict(state_next)[0]))
            q_values = self.modelnn.predict(state)
            q_values[0][action] = q_update
            self.modelnn.fit(state, q_values)

        self.calc_exploration()

    def calc_exploration(self):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_min, self.exploration_rate)

    def train(self):
        self.reset()
        score_logger = ScoreLogger(self.env_name)

        for run in range(self.max_iter):
            state = self.env.reset(self.num)
            if self.log_state:
                print("STATE")
                print(state)
            state = np.reshape(state, [1, self.observation_space])
            step = 0
            while self.num < self.max_examples:
                step += 1
                #self.env.render()
                action = self.act(state)
                state_next, reward, terminal, info = self.env.step(action, self.num)
                if self.log_state:
                    print(action, reward, terminal)
                #reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, self.observation_space])
                self.remember(state, action, reward, state_next, terminal)
                if self.num >= self.max_examples:
                    terminal = True
                state = state_next
                if terminal:
                    str = "Run: {:d}, exploration: {:.5f}, score: {:d}, memory: {:d}, balance: {:.2f}"
                    print(str.format(run+1, self.exploration_rate, step, len(self.memory), self.env.getBalance()))
                    score_logger.add_score(step, run+1)
                    break
                self.experience_replay()

            if self.num >= self.max_examples:
                break
