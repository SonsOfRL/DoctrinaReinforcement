
def test_gather_and_scatter():
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
        send_array = np.ones((200, 1), dtype=np.int32)
    else:
        send_array = np.ones((100, 4, 84, 84), dtype=np.uint8)

    recv = gather_and_scatter(
        local_comm, send_array, remote_comm, remote_leader)
    print(recv.shape, rank, send_array.shape)
