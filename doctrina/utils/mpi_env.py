import gym
from copy import deepcopy
from mpi4py import MPI
import numpy as np
import sys

from doctrina.utils.mpi_utils import list_gather_and_scatter, gather_and_scatter


class MpiEnvComm():
    """ Base communication class for MPI environments.
    """

    def __init__(self, n_env_proc):
        self.n_env_proc = n_env_proc
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()

        if size <= n_env_proc:
            raise ValueError(
                "Not enough processes for {} environments in {} processes"
                .format(n_env_proc, size))

        color = MpiEnvComm.get_color(n_env_proc)
        self.local_comm = MPI.COMM_WORLD.Split(color=color, key=rank)
        self.color = color
        remote_leader = 0 if color == 1 else size - n_env_proc

        self.remote_comm = self.local_comm.Create_intercomm(
            local_leader=0,
            peer_comm=MPI.COMM_WORLD,
            remote_leader=remote_leader,
            tag=12)

    @staticmethod
    def get_color(n_env_proc):
        size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
        return 1 if size - rank <= n_env_proc else 0

    def render(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def write_profile(pr):
        pr.disable()
        if MPI.COMM_WORLD.Get_rank() < 5:
            pr.dump_stats("logs/atari_cpu_%d.prof" % (MPI.COMM_WORLD.Get_rank()+1))
        # with open("cpu_%d.txt" % (MPI.COMM_WORLD.Get_rank()+1), "w") as output_file:
        #     sys.stdout = output_file
        #     pr.print_stats(sort="cumulative")
        #     sys.stdout = sys.__stdout__


class MpiEnvClient(MpiEnvComm):
    """ Parallel environment class that utilize MPI functionallity.
    """

    def __init__(self, nenv, n_env_proc, env_fn, pr, **env_kwargs):
        super().__init__(n_env_proc)
        self.nenv = nenv
        self.pr = pr
        pr.enable()

        env = env_fn(**env_kwargs)

        if not isinstance(env.action_space, gym.spaces.discrete.Discrete):
            raise ValueError("Only discrete actions are supported!")

        self.observation_space_shape = env.observation_space.shape
        self.observation_space_dtype = env.observation_space.dtype
        self.action_space_size = env.action_space.n
        self.action_space_dtype = env.action_space.dtype

    def send_cmd(self, command):
        """ Inform the worker(environment) processes to wait for a specific
        job. Available commands are as follows:
            - s_state: Used for observe method to inform environments that
            gpu processes will be asking for state information.
            - s_state_r_action: Receive actions from gpu and send states
            - s_trans_r_action: Receive actions from gpu and send transition
            triplet.
            - s_trans: Send transition triplet to gpu.
            - switch_envs: Tell environments to switch the second group of
            environments. Used for **pipelining**.
            - terminate: Terminate environment processes.
        Worker processes always wait for a command after performing a task.
        """
        cmd_list = ["s_state", "s_state_r_action", "s_trans_r_action",
                    "s_trans", "switch_envs", "r_action_s_trans", "terminate"]
        if isinstance(command, str):
            if command not in cmd_list:
                raise ValueError("Given string command is not available")
            command = cmd_list.index(command)
        self.remote_comm.bcast({"command": command}, MPI.ROOT)

    def observe(self, actions=None, swap=False):
        """ Gather and distribute states from environment to gpus nodes.
        If the "actions" is
        """
        if (self.nenv * self.n_env_proc) % self.local_comm.Get_size() != 0:
            raise RuntimeError(
                ("#environment per env processs times the #env process must"
                 "be an exact multiple of #gpu processes. {} x {} % {}")
                .format(self.nenv, self.n_env_proc, self.local_comm.Get_size()))

        if actions is not None:
            if len(actions.shape) != 2:
                raise ValueError("Action must be 2 dimensional. (B x k)")

        # Send empty buffer to environment processes
        if actions is None:
            sendbuf = np.empty(0)
            self.send_cmd("s_state")
        else:
            sendbuf = actions
            self.send_cmd("s_state_r_action")

        states = gather_and_scatter(self.local_comm, sendbuf, self.remote_comm)

        if swap:
            self.send_cmd("switch_envs")

        return states

    def step(self, actions=None, swap=False):
        """ Distribute actions to environments to step. Gather transition
        triplets.
        Arguments:
            - actions: (B x k) where k is the number of actions per sample.
            "k" is 1 for discrete action spaces.
        """

        if actions is not None:
            if len(actions.shape) != 2:
                raise ValueError("Action must be 2 dimensional. (B x k)")

        # Send action buffer to environment processes
        if actions is None:
            sendbuf = np.empty(0)
            self.send_cmd("s_trans")
        else:
            sendbuf = actions
            self.send_cmd("s_trans_r_action")

        transition_triplet = list_gather_and_scatter(
            self.local_comm, [sendbuf], self.remote_comm)

        if swap:
            self.send_cmd("switch_envs")
        next_state, reward, terminal = transition_triplet

        return next_state, reward, terminal

    def step_no_pipeline(self, actions):
        """ Distribute actions to environments to step. Gather transition
        triplets.
        Arguments:
            - actions: (B x k) where k is the number of actions per sample.
            "k" is 1 for discrete action spaces.
        """

        if len(actions.shape) != 2:
            raise ValueError("Action must be 2 dimensional. (B x k)")

        # Send action buffer to environment processes
        sendbuf = actions
        self.send_cmd("r_action_s_trans")
        gather_and_scatter(self.local_comm, sendbuf, self.remote_comm)

        sendbuf = np.empty(0)
        transition_triplet = list_gather_and_scatter(
            self.local_comm, [sendbuf], self.remote_comm)

        next_state, reward, terminal = transition_triplet
        return next_state, reward, terminal

    def __del__(self):
        self.send_cmd("terminate")


class MpiEnvWorker(MpiEnvComm):
    """ MPI worker class that runs the environments.
    """

    def __init__(self, nenv, n_env_proc, env_fn, pr, **env_kwargs):
        super().__init__(n_env_proc)
        self.nenv = nenv
        self.pr = pr
        self.pr.enable()

        # Initialize environments for per processes(double the given amount)
        self._envs = [{"env": env_fn(**env_kwargs),
                       "done": False}
                      for i in range(nenv*2)]

        self.observation_space = self._envs[0]["env"].observation_space
        self.action_space = self._envs[0]["env"].action_space
        self.active_slice = slice(0, nenv)

        # Reset all the environments & start worker loop
        self._states = np.stack([env["env"].reset() for env in self._envs])
        self.transition = None

        self.worker_loop()

    def step(self, actions):
        next_states, rewards, terminals, _ = [
            np.stack(x, axis=0) for x in
            zip(*[env.step(a.item()) for a, env in zip(actions, self.envs)])]

        # Reset the environments if necessary
        for ix, (done, n_state, env) in enumerate(zip(terminals,
                                                      next_states,
                                                      self.envs)):
            if done.item() is True:
                self.states[ix] = env.reset()
            else:
                self.states[ix] = n_state

        return next_states, rewards.astype(np.float64), terminals

    def _switch_envs(self):
        if self.active_slice.start == 0:
            self.active_slice = slice(self.nenv, self.nenv*2)
        else:
            self.active_slice = slice(0, self.nenv)

    def receive_cmd(self):
        """ Receive commands from gpu processes.
        Docstring for commands is available in MpiEnv class.
        """
        cmd = self.remote_comm.bcast(None, root=0)
        return cmd

    def _s_state(self):
        """ Send states of the last active environments.
        """
        gather_and_scatter(self.local_comm, self.states, self.remote_comm)

    def _r_action_s_transition(self):
        """ Receive action send state for none pipelining
        """
        sendbuf = np.empty(0)
        actions = gather_and_scatter(
            self.local_comm, sendbuf, self.remote_comm)
        transition = self.step(actions)
        list_gather_and_scatter(
            self.local_comm, transition, self.remote_comm)

    def _s_state_r_action(self):
        """
        """
        actions = gather_and_scatter(
            self.local_comm, self.states, self.remote_comm)
        next_states, rewards, terminals = self.step(actions)
        self.transition = (next_states, rewards, terminals)

    def _s_trans_r_action(self):
        """
        """
        actions = list_gather_and_scatter(
            self.local_comm, self.transition, self.remote_comm)[0]
        next_states, rewards, terminals = self.step(actions)
        self.transition = (next_states, rewards, terminals)

    def _s_trans(self):
        """
        """
        list_gather_and_scatter(
            self.local_comm, self.transition, self.remote_comm)

    def worker_loop(self):
        function_call_dict = {
            0: self._s_state,
            1: self._s_state_r_action,
            2: self._s_trans_r_action,
            3: self._s_trans,
            4: self._switch_envs,
            5: self._r_action_s_transition
        }
        while True:
            cmd = self.receive_cmd()
            command = cmd["command"]
            # Careful here!!!!!
            if command == 6:
                MpiEnvComm.write_profile(self.pr)
                exit()
            job = function_call_dict[command]
            job()

    @property
    def envs(self):
        return [env["env"] for env in self._envs[self.active_slice]]

    @property
    def states(self):
        return self._states[self.active_slice]


def MpiEnv(nenv, n_env_proc, env_fn, pr, **env_kwargs):
    """ Fake class
    """
    color = MpiEnvComm.get_color(n_env_proc)

    if color == 0:
        return MpiEnvClient(nenv, n_env_proc, env_fn, pr, **env_kwargs)
    else:
        return MpiEnvWorker(nenv, n_env_proc, env_fn, pr, **env_kwargs)


if __name__ == "__main__":
    env = MpiEnv(nenv=3, n_env_proc=1,
                 env_fn=lambda: gym.make("LunarLander-v2"))
    actions = np.arange(env.nenv * env.n_env_proc).reshape(-1, 1)

    for i in range(10000):
        states = env.observe(swap=False)
        # states = env.observe(actions)
        next_s, reward, terminal = env.step_no_pipeline(actions)
        # next_s, reward, terminal = env.step()
        # print(states, next_s, terminal, reward)

    env.send_cmd("terminate")
    # print(reward)
