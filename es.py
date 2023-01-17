import datetime
import os
import numpy as np
import scipy.stats as ss
import torch
import torch.multiprocessing as mp
import gym
from worker import Worker, WorkerInput, sample_noise
from network import ESPolicyNetwork
from plot import plot

# Params
MAX_WORKERS = 12 # Number of concurrent worker processes to use during training
ES_POP = 512 # Number of policy mutations to generate per iteration
ES_LR = 0.01 # Learning rate
ES_SIG = 0.1 # Mutation standard deviation
MAX_I = 10000 # Maximum number of training iterations
TEST_E = 10 # Number of testing episodes over which to get average performance
TEST_F = 10 # Interval between tests

ENV_NAME = "Acrobot-v1" # Gym environment to train on
ENV_TRUNC = 500 # Max episode length
REW_THR = 0 # Average test reward threshold at which to stop training

# Globals
out_dir = "./runs/"

def main():
    # Logging
    global out_dir
    now = datetime.datetime.now()
    out_dir += now.strftime("%y%m%d%H%M%S") + "/"
    os.mkdir(out_dir)
    with open(out_dir + "log_train.txt", "a") as f:
        f.write("{},{},{},{}\n".format("i", "rew_avg", "rew_min", "rew_max"))
    with open(out_dir + "log_test.txt", "a") as f:
        f.write("{},{}\n".format("i", "rew_avg"))

    # Test environment
    test_env = gym.make(ENV_NAME)

    # Agent network and optimiser
    net = ESPolicyNetwork(test_env.observation_space.shape, test_env.action_space.n)
    optimiser = torch.optim.Adam(net.parameters(), lr = ES_LR)

    # Workers
    input_queue = mp.Queue(maxsize = ES_POP)
    output_queue = mp.Queue(maxsize = ES_POP)
    workers = []
    for _ in range(MAX_WORKERS):
        worker = Worker(input_queue, output_queue, ENV_NAME, ENV_TRUNC)
        worker.start()
        workers.append(worker)

    # Training loop
    for i in range(MAX_I):
        print("\n--- {} ---".format(i))

        # Queue up inputs for workers
        policy_params = net.state_dict()
        for _ in range(ES_POP):
            input = WorkerInput(policy_params, ES_SIG)
            input_queue.put(input)
        
        # Get seeds and rewards from workers
        batch_noise = []
        batch_reward = []
        for _ in range(ES_POP):
            output = output_queue.get()
            np.random.seed(output.seed)
            noise = sample_noise(net)
            batch_noise.append(noise)
            batch_reward.append(output.total_reward)
    
        reward_avg = np.mean(batch_reward)
        reward_min = np.min(batch_reward)
        reward_max = np.max(batch_reward)

        # Normalise rewards
        ranked = ss.rankdata(batch_reward)
        norm_reward = (ranked - 1) / (len(ranked) - 1)
        norm_reward -= 0.5

        # Calculate updated network parameters
        optimiser.zero_grad()
        for idx, p in enumerate(net.parameters()):
            upd_weights = np.zeros(p.data.shape)

            for n, r in zip(batch_noise, norm_reward):
                upd_weights += r * n[idx]

            upd_weights /= (ES_POP * ES_SIG)
            
            # Set parameter gradient for optimiser
            p.grad = torch.FloatTensor(-upd_weights)

        # Optimise
        optimiser.step()

        # Logging
        print("Training Reward Avg: {}, Min: {}, Max: {}".format(reward_avg, reward_min, reward_max))
        with open(out_dir + "log_train.txt", "a") as f:
            f.write("{},{},{},{}\n".format(i, reward_avg, reward_min, reward_max))

        # Testing
        if i % TEST_F == 0:
            test_reward = test(net, TEST_E, test_env)
            print("Test Reward Avg: {}".format(test_reward))
            with open(out_dir + "log_test.txt", "a") as f:
                f.write("{},{}\n".format(i, test_reward))

            plot(out_dir, ENV_NAME)
            
            if test_reward > REW_THR:
                print("Training complete!")
                torch.save(net.state_dict(), out_dir + "complete.pt")
                exit()

    print("Max training steps reached.")
    exit()

def test(net, eps, env):
    rewards = []
    for e in range(eps):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(ENV_TRUNC):
            dist = net(torch.tensor(state))
            action = torch.argmax(dist.probs).item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if done: break
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)

if __name__ == '__main__':
    main()