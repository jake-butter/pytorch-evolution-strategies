import datetime
import os

import gymnasium as gym
import numpy as np
import scipy.stats as ss
import torch
import torch.multiprocessing as mp
import typer

from network import ESPolicyNetwork
from plot import plot
from worker import Worker, WorkerInput, sample_noise

# Globals
out_dir = "./runs/"


def main(
    max_workers: int = 12,
    es_pop: int = 512,
    es_lr: float = 0.01,
    es_sig: float = 0.1,
    max_i: int = 10000,
    test_e: int = 10,
    test_f: int = 10,
    env_name: str = "Acrobot-v1",
    env_trunc: int = 500,
    rew_thr: float = 0.0,
):
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
    test_env = gym.make(env_name)

    # Agent network and optimiser
    net = ESPolicyNetwork(test_env.observation_space.shape, test_env.action_space.n)
    optimiser = torch.optim.Adam(net.parameters(), lr=es_lr)

    # Workers
    input_queue = mp.Queue(maxsize=es_pop)
    output_queue = mp.Queue(maxsize=es_pop)
    workers = []
    for _ in range(max_workers):
        worker = Worker(input_queue, output_queue, env_name, env_trunc)
        worker.start()
        workers.append(worker)

    # Training loop
    for i in range(max_i):
        print("\n--- {} ---".format(i))

        # Queue up inputs for workers
        policy_params = net.state_dict()
        for _ in range(es_pop):
            input = WorkerInput(policy_params, es_sig)
            input_queue.put(input)

        # Get seeds and rewards from workers
        batch_noise = []
        batch_reward = []
        for _ in range(es_pop):
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

            upd_weights /= es_pop * es_sig

            # Set parameter gradient for optimiser
            p.grad = torch.FloatTensor(-upd_weights)

        # Optimise
        optimiser.step()

        # Logging
        print(
            "Training Reward Avg: {}, Min: {}, Max: {}".format(
                reward_avg, reward_min, reward_max
            )
        )
        with open(out_dir + "log_train.txt", "a") as f:
            f.write("{},{},{},{}\n".format(i, reward_avg, reward_min, reward_max))

        # Testing
        if i % test_f == 0:
            test_reward = test(net, test_e, test_env, env_trunc)
            print("Test Reward Avg: {}".format(test_reward))
            with open(out_dir + "log_test.txt", "a") as f:
                f.write("{},{}\n".format(i, test_reward))

            plot(out_dir, env_name)

            if test_reward > rew_thr:
                print("Training complete!")
                torch.save(net.state_dict(), out_dir + "complete.pt")
                exit()

    print("Max training steps reached.")
    exit()


def test(net, eps, env, env_trunc):
    rewards = []
    for e in range(eps):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(env_trunc):
            dist = net(torch.tensor(state))
            action = torch.argmax(dist.probs).item()
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)

    return np.mean(rewards)


if __name__ == "__main__":
    typer.run(main)
