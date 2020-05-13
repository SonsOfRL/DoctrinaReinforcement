import torch
import numpy as np
import gym
import argparse
import cProfile

from doctrina.agents.a2c.model import A2C
from doctrina.agents.a2c.training import train
from doctrina.utils.mpi_env import MpiEnv, MpiEnvComm
from doctrina.utils.atari_wrapper import ResizeAndShape
from doctrina.utils.writers import PrintWriter


class Net(torch.nn.Module):

    def __init__(self, channels, num_actions, memsize=128):
        super().__init__()
        self.memsize = memsize
        self.conv1 = torch.nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = torch.nn.GRUCell(32 * 5 * 5, memsize)
        self.critic_linear = torch.nn.Linear(memsize, 1)
        self.actor_linear = torch.nn.Linear(memsize, num_actions)

    def forward(self, x, hx):
        x = torch.nn.functional.elu(self.conv1(x))
        x = torch.nn.functional.elu(self.conv2(x))
        x = torch.nn.functional.elu(self.conv3(x))
        x = torch.nn.functional.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return self.actor_linear(hx), self.critic_linear(hx), hx

    def reset_hx(self, bs):
        return torch.zeros(bs, self.memsize)


def makeenv(envname):
    env = gym.make(envname)
    return ResizeAndShape(env)


def main(args):
    # Before agent initialization
    pr = cProfile.Profile()

    mpienv = MpiEnv(args.nenv_per_core,
                    args.nenv_proc,
                    lambda: makeenv(args.envname),
                    pr=pr)

    env = makeenv(args.envname)
    in_size = env.observation_space.shape[-1]
    out_size = env.action_space.n
    network = Net(in_size, out_size)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    agent = A2C(network, optimizer)
    agent.to(args.device)
    del env

    writer = PrintWriter(flush=True)

    train(args, mpienv, agent, writer)
    MpiEnvComm.write_profile(mpienv.pr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Mnist')
    parser.add_argument("--envname", type=str,
                        default="Pong-v0",
                        help="Name of the environment")
    parser.add_argument("--nenv-per-core", type=int,
                        help="Number of environemnts run in parallel",
                        default=1)
    parser.add_argument("--nenv-proc", type=int,
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
                        default=0.3)
    parser.add_argument("--write-period", type=int,
                        help="Logging period",
                        default=100)
    args = parser.parse_args()
    main(args)
