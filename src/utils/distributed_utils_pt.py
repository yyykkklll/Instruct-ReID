import os

import torch
import torch.distributed as dist
import subprocess
import socket
import time
from . import comm_
import torch.cuda.comm


def find_free_port():
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]

def dist_init():
    
    hostname = socket.gethostname()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if int(os.environ["RANK"]) == 0:
            print('this task is not running on cluster!')
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        dist_url = 'env://'
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
        addr = socket.gethostname()

    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        if proc_id == 0:
            print('Init dist using slurm!')
            print("Job Id is {} on {} ".format(os.environ["SLURM_JOBID"], os.environ['SLURM_NODELIST']))
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "dist_url_" + jobid + ".txt"
        if proc_id == 0:
            tcp_port = str(find_free_port())
            print('write port {} to file: {} '.format(tcp_port, hostfile))
            with open(hostfile, "w") as f:
                f.write(tcp_port)
        else:
            print('read port from file: {}'.format(hostfile))
            while not os.path.exists(hostfile):
                time.sleep(1)
            time.sleep(2)
            with open(hostfile, "r") as f:
                tcp_port = f.read()

        os.environ['MASTER_PORT'] = str(tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        dist_url = 'env://'
        world_size = ntasks
        rank = proc_id
        gpu = proc_id % num_gpus
    else:
        print('Not using distributed mode')
        distributed = False
        return
    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    # rank = int(os.environ["RANK"])
    # world_size = int(os.environ['WORLD_SIZE'])
    print('rank: {} addr: {}  port: {}'.format(rank, addr, os.environ['MASTER_PORT']))
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    if 'SLURM_PROCID' in os.environ and rank == 0:
        if os.path.isfile(hostfile):
            os.remove(hostfile)
    if world_size >= 1:
        # Setup the local process group (which contains ranks within the same machine)
        assert comm_._LOCAL_PROCESS_GROUP is None
        num_gpus = torch.cuda.device_count()
        num_machines = world_size // num_gpus
        for i in range(num_machines):
            ranks_on_i = list(range(i * num_gpus, (i + 1) * num_gpus))
            print('new_group: {}'.format(ranks_on_i))
            pg = torch.distributed.new_group(ranks_on_i)
            if rank in ranks_on_i:
                # if i == os.environ['SLURM_NODEID']:
                comm_._LOCAL_PROCESS_GROUP = pg
    return rank, world_size

def dist_init_singletask(args):
    os.environ['TORCH_DISTRIBUTED_DISABLE_LIBUV'] = '1'

    # 检查是否为分布式训练
    if args.local_rank is not None and args.local_rank >= 0:  # 分布式训练
        # 设置 MASTER_ADDR 和 MASTER_PORT
        os.environ['MASTER_ADDR'] = 'localhost'  # 在本地运行时使用 localhost
        os.environ['MASTER_PORT'] = str(args.port) if hasattr(args, 'port') else '29500'

        # 设置 RANK 和 WORLD_SIZE（如果未设置）
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(args.local_rank)
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

        # 初始化分布式训练
        dist.init_process_group(backend='gloo', init_method='env://')  # 使用 gloo 后端（Windows 支持）
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_distributed = True
        print(f"Initialized distributed training: rank {rank}/{world_size}")
    else:
        # 单进程运行
        rank = 0
        world_size = 1
        is_distributed = False
        print("Running in single-process mode")

    return rank, world_size, is_distributed
