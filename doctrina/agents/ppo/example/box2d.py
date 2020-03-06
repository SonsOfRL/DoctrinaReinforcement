import torch
import numpy as np
import gym
import argparse

from doctrina.agents.ppo.model import PPO
from doctrina.agents.ppo.training import train
from doctrina.utils.parallel_envs import ParallelEnv


class Net(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        self.policynet = torch.nn.Sequential(
            torch.nn.Linear(in_size, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_size)
        )

        self.valuenet = torch.nn.Sequential(
            torch.nn.Linear(in_size, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        gain = torch.nn.init.calculate_gain("relu")

        def param_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight, gain)
                torch.nn.init.zeros_(module.bias)
        self.apply(param_init)

    def forward(self, state):
        value = self.valuenet(state)
        logits = self.policynet(state)

        return logits, value


def main(args):

    env = gym.make(args.envname)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n
    network = Net(in_size, out_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    agent = PPO(network, optimizer)
    agent.to(args.device)
    del env

    penv = ParallelEnv(args.nenv, lambda: gym.make(args.envname))
    train(args, penv, agent, print)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mnist')
    parser.add_argument("--envname", type=str,
                        default="LunarLander-v2",
                        help="Name of the environment")
    parser.add_argument("--nenv", type=int,
                        help="Number of environemnts run in parallel",
                        default=16)
    parser.add_argument("--lr", type=float, help="Learning rate", default=3e-4)
    parser.add_argument("--device", type=str, help="Torch device",
                        default="cuda")
    parser.add_argument("--n-iter", type=int,
                        help="Number of iterations",
                        default=int(1e4))
    parser.add_argument("--n-step", type=int,
                        help="Number of iterations",
                        default=20)
    parser.add_argument("--gamma", type=float,
                        help="Discount factor",
                        default=0.98)
    parser.add_argument("--beta", type=float,
                        help="Entropy coefficient",
                        default=0.1)
    parser.add_argument("--write-period", type=int,
                        help="Logging period",
                        default=100)
    parser.add_argument("--epsilon", type=float,
                        help="Clip Ratio",
                        default=0.2)
    args = parser.parse_args()
    main(args)
