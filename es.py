import datetime
import os
from collections import namedtuple
import numpy as np
import torch
import gym

# Params
MAX_WORKERS = 128
ES_POP = 512
ES_LR = 0.01
ES_SIG = 0.1

# Globals
out_dir = "./runs/"


def main():
    # Logging
    global out_dir
    now = datetime.datetime.now()
    out_dir += now.strftime("%y%m%d%H%M%S") + "/"
    os.mkdir(out_dir)
    with open(out_dir + "log_train.txt", "a") as f:
        f.write("{},{},{},{},{}\n".format("e", "n", "rew_avg", "rew_min", "rew_max"))
    os.mkdir(out_dir + "checkpoints/")

    # Test environment
    test_env = CartPoleFlip()

    # Agent
    agent = Agent(test_env.observation_space.shape, test_env.action_space.n, SUB_POP, SEL_POP, MAX_WORKERS, SUB_N, SUB_LR, SUB_SIG, SEL_LR, SEL_SIG)
    for _ in range(SUB_N):
        agent.add_policy()

    e = 0
    while True:
        print("\n--- {} ---".format(e))

        # Optimise each subpolicy
        for n in range(SUB_N):
            print("Training subpolicy {}...".format(n))
            # Step
            rew_avg, rew_min, rew_max, rew_avg_reg, rew_min_reg, rew_max_reg, rew_avg_flip, rew_min_flip, rew_max_flip, sel_avg_reg, sel_avg_flip = agent.es_step(n)
            
            # Logging
            print("Reward Avg: {}, Max: {}, Diff: {}".format(rew_avg, rew_max, rew_avg_reg - rew_avg_flip))
            with open(out_dir + "log_train.txt", "a") as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(e, n, rew_avg, rew_min, rew_max, rew_avg_reg, rew_min_reg, rew_max_reg, rew_avg_flip, rew_min_flip, rew_max_flip, sel_avg_reg, sel_avg_flip))
        
        # Optimiser selector
        print("///")
        print("Training critic...")
        rew_avg, rew_min, rew_max, err_avg, err_min, err_max, sel_avg_reg, sel_min_reg, sel_max_reg, sel_avg_flip, sel_min_flip, sel_max_flip = agent.es_step_selector()

        # Logging
        print("Reward Avg: {}, Min: {}, Max: {}".format(rew_avg, rew_min, rew_max))
        print("MSE Avg: {}, Min: {}, Max: {}".format(err_avg, err_min, err_max))
        print("Selection Regular: {}, Flipped: {}".format(sel_avg_reg, sel_avg_flip))
        with open(out_dir + "log_train_selector.txt", "a") as f:
            f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(e, rew_avg, rew_min, rew_max, err_avg, err_min, err_max, sel_avg_reg, sel_min_reg, sel_max_reg, sel_avg_flip, sel_min_flip, sel_max_flip))
        
        # Plotting
        if e % 10 == 0:
            plot(out_dir, SUB_N)
        
        e += 1

if __name__ == '__main__':
    main()

def es_step(self, sub_n):
        # Queue up network params, sigs
        subpolicy_params = self.sub_policies[sub_n].get_state_dict()
        for _ in range(self.subpolicy_pop):
            input = WorkerInput(subpolicy_params, self.sub_policies[sub_n].sigma, self.selector.state_dict(), sub_n)
            self.input_queue.put(input)

        # Get rewards and seeds from workers
        batch_noise = []
        batch_reward = []
        batch_env_flipped = []
        batch_sels = []
        batch_sel_vecs = []
        for _ in range(self.subpolicy_pop):
            output = self.output_queue.get()
            np.random.seed(output.seed)
            noise = sample_noise(self.sub_policies[sub_n].policy)
            batch_noise.append(noise)
            batch_reward.append(output.total_reward)
            batch_env_flipped.append(output.flipped)
            batch_sels.append(output.sel_avg)
            batch_sel_vecs.append(output.sel_vec)
        
        reward_avg = np.mean(batch_reward)
        reward_max = np.max(batch_reward)
        reward_min = np.min(batch_reward)

        # Average by env type
        rewards_reg = []
        rewards_flip = []
        sels_reg = []
        sels_flip = []
        for i, flipped in enumerate(batch_env_flipped):
            if not flipped:
                rewards_reg.append(batch_reward[i])
                sels_reg.append(batch_sels[i])
            else:
                rewards_flip.append(batch_reward[i])
                sels_flip.append(batch_sels[i])
        
        reward_avg_reg = np.mean(rewards_reg)
        reward_max_reg = np.max(rewards_reg)
        reward_min_reg = np.min(rewards_reg)
        reward_avg_flip = np.mean(rewards_flip)
        reward_max_flip = np.max(rewards_flip)
        reward_min_flip = np.min(rewards_flip)

        sel_avg_reg = np.mean(sels_reg)
        sel_avg_flip = np.mean(sels_flip)

        sel_centre = np.mean(batch_sel_vecs, 0)[sub_n]
        trimmed_rewards = []
        trimmed_noise = []
        for i, sel_vec in enumerate(batch_sel_vecs):
            if sel_vec[sub_n] > sel_centre:
                trimmed_rewards.append(batch_reward[i])
                trimmed_noise.append(batch_noise[i])

        norm_trimmed = self.normalised_rank(trimmed_rewards)

        # Calculate updated network parameters
        self.sub_policies[sub_n].optimizer.zero_grad()
        for idx, p in enumerate(self.sub_policies[sub_n].policy.parameters()):
            upd_weights = np.zeros(p.data.shape)

            for n, r in zip(trimmed_noise, norm_trimmed):
                upd_weights += r * n[idx]

            upd_weights /= (self.subpolicy_pop * self.sub_policies[sub_n].sigma)
            
            # Set parameter gradient for optimizer
            p.grad = torch.FloatTensor(-upd_weights)

        # Optimize
        self.sub_policies[sub_n].optimizer.step()
        
        return reward_avg, reward_min, reward_max, reward_avg_reg, reward_min_reg, reward_max_reg, reward_avg_flip, reward_min_flip, reward_max_flip, sel_avg_reg, sel_avg_flip