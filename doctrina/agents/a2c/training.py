import torch
import numpy as np


def train(args, penv, agent, writer=None):
    eps_rewards = np.zeros((args.nenv, 1))
    reward_list = [0]
    loss_list = [0]

    def to_torch(array):
        if len(array.shape) == 4:
            array = np.transpose(array, (0, 3, 1, 2))
        return torch.from_numpy(array).to(args.device).float()

    with penv as state:
        state = to_torch(state)
        for i in range(0, args.n_iter, args.n_step):
            for j in range(args.n_step):
                action, log_prob, value, entropy = agent(state)
                action = action.unsqueeze(1).cpu().numpy()
                next_state, reward, done = penv.step(action)
                next_state = to_torch(next_state)
                with torch.no_grad():
                    _, next_value = agent.network(next_state)
                agent.add_trans(to_torch(reward), to_torch(done),
                                log_prob.unsqueeze(1), value,
                                next_value, entropy)
                state = next_state
                for k, d in enumerate(done.flatten()):
                    eps_rewards[k] += reward[k].item()
                    if d == 1:
                        reward_list.append(eps_rewards[k].item())
                        eps_rewards[k] = 0
                if writer and (i + j) % args.write_period == 0:
                    writer(("Iteration: {}, Reward: {}, Loss: {}")
                           .format(i + j,
                                   np.mean(reward_list[-100:]),
                                   np.mean(loss_list[-10:])))
            loss_list.append(agent.update(gamma=args.gamma, beta=args.beta) / args.n_step)
            