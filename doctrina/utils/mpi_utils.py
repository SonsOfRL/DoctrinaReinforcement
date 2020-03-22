from mpi4py import MPI
import numpy as np


def gather_and_scatter(local_comm, send_array, remote_comm):
    """ Combined MPI function that works similar to alltoall function. Gather
    the buffers locally and scatter them uniformly through an
    inter-communicater. This functionality is only served for two groups.

    process | Group A | Group B | Local Root | Remote Root | Gather  | Scatter
    --------|---------|---------|------------|-------------|---------|--------
       A    |    +    |         |     A      |      C      |  A, B   |   C,D
       B    |    +    |         |     A      |      C      |   --    |   E,F
       C    |         |    +    |     C      |      A      | C,D,E,F | A[:HALF]
       D    |         |    +    |     C      |      A      |   --    | A[HALF:]
       E    |         |    +    |     C      |      A      |   --    | B[:HALF]
       F    |         |    +    |     C      |      A      |   --    | B[HALF:]

    An example shown above shows the collective behaviuor of the function. MPI
    calls used in this function are as follows:
    - bcast: Remote leaders broadcast the shape, #processes and dtype
    informations to the other group.
    - Gather: Local leaders(rank 0 in local_comm) gather buffers locally
    - Scatter: Remote leaders scatter the gathered data by uniformly dividing
    it to remote processes. Note that, gathered buffer must be divedable to
    # remote processes.

    Arguments:
        - local_comm: MPI communicater instance for the local group
        - send_array: Array to be sent through remote communicator
        - remote_comm: MPI inter-communicator instance(from group A to B or
        vise-versa)

    Return:
        Buffer that is recieved from remote processes (Dtype is determined by
        remote processes). Size of the data is equally divided among receiving
        processes.

    Raise:
        - ValueError: If the gathered data cannot be equally divided.
    """
    # Gather data from local_comm
    recvbuf = None
    local_leader = 0
    local_rank = local_comm.Get_rank()
    # <-------------- Get shape and dtype of remote ------------->
    # Local leader
    if local_rank == local_leader:
        info = {"dsize": len(send_array.flatten()),
                "dtype": send_array.dtype,
                "shape": send_array.shape,
                "size": local_comm.Get_size()}
        remote_comm.bcast(info, MPI.ROOT)
    recvbuf_info = remote_comm.bcast(None, root=local_leader)

    if (len(send_array.flatten()) * local_comm.Get_size() %
            recvbuf_info["size"] != 0):
        raise ValueError("Gathered data cannot be equally divided")

    # <-------------- Gather send_array's to local leader ------------->
    # Local leader
    if local_rank == local_leader:
        send_shape = send_array.shape
        local_proc_size = local_comm.Get_size()
        recvbuf = np.empty((local_proc_size, *send_shape),
                           dtype=send_array.dtype)

    # Local leader gathers the data from local processes of the group
    local_comm.Gather(send_array, recvbuf, root=local_leader)

    # Local leader

    if local_rank == local_leader:
        sendbuf = recvbuf
        remote_comm.Scatter(sendbuf.flatten(), None, root=MPI.ROOT)

    recvbuf = np.empty(
        shape=((recvbuf_info["dsize"] *
                recvbuf_info["size"]) // local_comm.Get_size()),
        dtype=recvbuf_info["dtype"])
    remote_comm.Scatter(None, recvbuf, root=0)
    return recvbuf.reshape(-1, *(recvbuf_info["shape"][1:]))


def list_gather_and_scatter(local_comm, send_list, remote_comm):
    """
    """
    if not isinstance(send_list, (tuple, list)):
        raise ValueError("Send list must be a list or tuple. Currently {}"
                         .format(type(send_list)))
    # Gather data from local_comm
    recv_list = [None] * len(send_list)
    local_leader = 0
    local_rank = local_comm.Get_rank()
    # <-------------- Get shape and dtype of remote ------------->
    # Local leader
    if local_rank == local_leader:
        info = {"size": local_comm.Get_size(),
                "list": [{"dtype": arr.dtype,
                          "dsize": np.product(arr.shape),
                          "shape": arr.shape,
                          } for arr in send_list]}
        remote_comm.bcast(info, MPI.ROOT)
    recvbuf_info = remote_comm.bcast(None, root=local_leader)

    # Check for distributability
    for arr in send_list:
        if (np.product(arr.shape) * local_comm.Get_size() %
                recvbuf_info["size"] != 0):
            raise ValueError("Gathered data cannot be equally divided")

    # <-------------- Gather send_list's to local leader ------------->
    # Local leader
    if local_rank == local_leader:
        recv_list = []
        for arr in send_list:
            send_shape = arr.shape
            local_proc_size = local_comm.Get_size()
            recvbuf = np.empty((local_proc_size, *send_shape),
                               dtype=arr.dtype)
            recv_list.append(recvbuf)

    # Local leader gathers the data from local processes of the group
    for (r_arr, s_arr) in zip(recv_list, send_list):
        local_comm.Gather(s_arr, r_arr, root=local_leader)

    # Scatter gathered arrays
    if local_rank == local_leader:
        send_list = recv_list

    # Local leader
    if local_rank == local_leader:
        for ix, s_arr in enumerate(recv_list):
            remote_comm.Scatter(s_arr.flatten(), None, root=MPI.ROOT)

    recv_list = []
    for ix, arr_info in enumerate(recvbuf_info["list"]):
        recvbuf = np.empty(
            shape=((arr_info["dsize"] *
                    recvbuf_info["size"]) // local_comm.Get_size()),
            dtype=arr_info["dtype"]
            )
        remote_comm.Scatter(None, recvbuf, root=0)
        recv_list.append(
            recvbuf.reshape(-1, *(arr_info["shape"][1:]))
        )

    return recv_list


def share_gradients(agent):
    """ Gradients are shared among gpu processes.
    """
    raise NotImplementedError
    sendbuf = torch.cat([p.grad for p in agent.parameters()
                         if p.grad is not None])
    recvbuf = torch.empty_like(sendbuf)
    torch.distributed.all_reduce_multigpu([sendbuf, recvbuf], op=ReduceOp.SUM)
    # Assign gradients
    grad_sizes = [np.product(p.shape) for p in p.parameters()
                  if p.grad is not None]
    summed_grads = recvbuf.split(grad_size)
    for p, grad in zip((p for p in agent.parameters() if p.grad is not None),
                       summed_grad):
        p.grad = summed_grads
    raise RuntimeError("summed_grads must be divided by #gpus")


if __name__ == "__main__":

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    color = rank >= (size // 3)
    local_comm = MPI.COMM_WORLD.Split(
        color=color,
        key=rank
    )

    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()
    remote_comm = local_comm.Create_intercomm(
        local_leader=0,
        peer_comm=MPI.COMM_WORLD,
        remote_leader=0 if color == 1 else (size // 3),
        tag=5
    )

    remote_leader = 0 if color == 1 else (size // 3),

    if color == 0:
        send_list = [np.ones((200, 1), dtype=np.int32)]
    else:
        send_list = [
            np.ones((100, 4, 84, 84), dtype=np.uint8),
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.int32),
        ]

    recv = list_gather_and_scatter(
        local_comm, send_list, remote_comm)
    print([x.shape for x in recv], [x.shape for x in send_list], rank, remote_leader)
