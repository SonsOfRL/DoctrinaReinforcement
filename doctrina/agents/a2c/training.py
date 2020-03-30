import torch
import numpy as np
from datetime import datetime


def train(args, mpienv, agent, writer=None):
    eps_rewards = np.zeros((args.nenv_per_core * args.nenv_proc, 1))
    reward_list = [0]
    loss_list = [0]

    # For Recurrent Networks
    try:
        model_hx = agent.network.reset_hx(eps_rewards.shape[0]).to(args.device)
    except AttributeError:
        model_hx = torch.zeros(eps_rewards.shape[0], 1).to(args.device)

    def to_torch(array):
        if len(array.shape) == 4:
            array = np.transpose(array, (0, 3, 1, 2))
        return torch.from_numpy(array).to(args.device).float()

    for i in range(0, args.n_iter, args.n_step):
        model_hx = model_hx.detach()
        for j in range(args.n_step):
            state = mpienv.observe(swap=False)
            state = to_torch(state)
            action, log_prob, value, entropy, *model_hx = agent(
                state, model_hx
            )
            action = action.unsqueeze(1).cpu().numpy()

            next_state, reward, done = mpienv.step_no_pipeline(action)
            reward = to_torch(reward.reshape(-1, 1))
            done = to_torch(done.reshape(-1, 1))

            # For Recurrent Networks
            if model_hx:
                model_hx = (1 - done) * model_hx[0]
            else:
                model_hx = torch.zeros(eps_rewards.shape[0], 1).to(args.device)

            next_state = to_torch(next_state)
            with torch.no_grad():
                _, next_value, *_ = agent.network(next_state, model_hx)
            agent.add_trans(reward, done,
                            log_prob.unsqueeze(1), value,
                            next_value, entropy)
            for k, d in enumerate(done.flatten()):
                eps_rewards[k] += reward[k].item()
                if d == 1:
                    reward_list.append(eps_rewards[k].item())
                    eps_rewards[k] = 0
            if writer and (i + j) % args.write_period == 0:
                writer({
                    "Iteration: {:8}": i + j,
                    "Reward: {:3.2f}": np.mean(reward_list[-args.write_period:]),
                    "loss: {:5.3f}": np.mean(loss_list[-(args.write_period//args.n_step):]),
                    "Time: {}": str(datetime.now()),
                })
        loss_list.append(agent.update(gamma=args.gamma,
                                      beta=args.beta) / args.n_step)
