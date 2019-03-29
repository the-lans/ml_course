import csv
from scores.dqnsolver import *
from scores.cartpole_gym import *
from scores.cartpole import *

if __name__ == "__main__":
    env = CartPole()
    dqn_solver = DQNSolver(env, ENV_NAME,
                           MAX_ITERATION,
                           MAX_EXAMPLES,
                           LEARNING_RATE,
                           MEMORY_SIZE,
                           BATCH_SIZE,
                           GAMMA,
                           EXPLORATION_MIN,
                           EXPLORATION_MAX,
                           EXPLORATION_DECAY,
                           RND_MEMORY,
                           LOG_STATE)
    dqn_solver.train()

    if env.log:
        with open(FILE_NAME_LOG, 'wt') as csvfile:
            csvhead = env.log[0].keys()
            cout = csv.DictWriter(csvfile, csvhead, delimiter=';', lineterminator='\n')
            cout.writeheader()
            cout.writerows(env.log)
        print("Write log >> " + FILE_NAME_LOG)
