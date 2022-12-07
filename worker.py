import torch

WorkerInput = namedtuple("WorkerInput", ("subpolicy_params", "sigma", "selector_params", "n"))
WorkerOutput = namedtuple("WorkerOutput", ("total_reward", "seed", "flipped", "sel_avg", "sel_vec"))

class Worker(mp.Process):
    def __init__(self, input_queue, output_queue):
        super(Worker, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.env = CartPoleFlip()
        self.local_subpolicy = SubPolicyNetwork(self.env.observation_space.shape, self.env.action_space.n)
    
    def run(self):
        while True:
            input = self.input_queue.get()

            if input != None:
                # Load network params
                self.local_subpolicy.load_state_dict(input.subpolicy_params)

                # Sample noise using a random seed
                seed = np.random.randint(1e6)
                np.random.seed(seed)
                noise = self.sample_noise(self.local_subpolicy)

                # Apply noise to worker subpolicy network
                for n, p in zip(noise, self.local_subpolicy.parameters()):
                    p.data += torch.FloatTensor(n * input.sigma)
                
                # Evaluate noisy subpolicy
                state = self.env.reset()
                episode_reward = 0

                for t in range(500):
                    dist = self.local_subpolicy(torch.tensor(state))
                    action = dist.sample().item()
                    next_state, reward, done, _ = self.env.step(action)
                    state = next_state
                    episode_reward += reward
                    if done: break

                output = WorkerOutput(episode_reward, seed)
                self.output_queue.put(output)
            
            else: break
    
    def sample_noise(self, net):
        # Sample noise for each parameter of the provided network
        nn_noise = []
        for n in net.parameters():
            noise = np.random.normal(size=n.data.numpy().shape)
            nn_noise.append(noise)
        return np.array(nn_noise, dtype=object)