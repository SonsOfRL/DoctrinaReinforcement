from mpi4py import MPI
import numpy as np


def gather_and_scatter(local_comm, send_array, remote_comm):
    """ Combined MPI function that works similar to alltoall function. Gather
    the buffers locally and scatter them uniformly through an
    inter-communicater. This functinality is only served for two groups.

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
    - Gather: Local leaders(rank 0 in local_comm) gathers buffers locally
    - Scatter: Remote leaders scatter the gathered data by uniformly dividing
    it to remote processes. Note that, gathered buffer must be divedable to
    #remote processes.

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
        info = {"shape": len(send_array.flatten()),
                "dtype": send_array.dtype,
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
        shape=((recvbuf_info["shape"] *
                recvbuf_info["size"]) // local_comm.Get_size()),
        dtype=recvbuf_info["dtype"])
    remote_comm.Scatter(None, recvbuf, root=0)
    return recvbuf


if __name__ == "__main__":
    pass
