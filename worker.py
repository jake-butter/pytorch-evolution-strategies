from collections import namedtuple
import torch
import torch.multiprocessing as mp
import numpy as np
import gymnasium as gym
from network import ESPolicyNetwork

WorkerInput = namedtuple("WorkerInput", ("policy_params", "sigma"))
WorkerOutput = namedtuple("WorkerOutput", ("total_reward", "seed"))

class Worker(mp.Process):
    def __init__(self, input_queue, output_queue, env_name, env_trunc):
        super(Worker, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.env = gym.make(env_name)
        self.env_trunc = env_trunc
        self.local_policy = ESPolicyNetwork(self.env.observation_space.shape, self.env.action_space.n)
    
    def run(self):
        while True:
            input = self.input_queue.get()

            if input != None:
                # Load network params
                self.local_policy.load_state_dict(input.policy_params)

                # Apply noise to worker subpolicy network
                seed = np.random.randint(1e6)
                np.random.seed(seed)
                for p in self.local_policy.parameters():
                    n = np.random.normal(size = p.data.numpy().shape)
                    p.data += torch.FloatTensor(n * input.sigma)
                
                # Evaluate noisy subpolicy
                episode_reward = 0
                state, _ = self.env.reset()
                for t in range(self.env_trunc):
                    dist = self.local_policy(torch.tensor(state))
                    action = dist.sample().item()
                    next_state, reward, done, _, _ = self.env.step(action)
                    state = next_state
                    episode_reward += reward
                    if done: break

                output = WorkerOutput(episode_reward, seed)
                self.output_queue.put(output)
            
            else: break