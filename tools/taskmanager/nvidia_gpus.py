import subprocess
import numpy as np

def get_num_of_gpus():
    num_of_gpus_command = 'nvidia-smi -L | wc -l'
    num_of_gpus = subprocess.getoutput(num_of_gpus_command)
    return int(num_of_gpus)


def get_memories():
    available_memories_command = 'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free'
    total_memories_command = 'nvidia-smi -q -d Memory | grep -A4 GPU | grep Total'
    available_memories = subprocess.getoutput(available_memories_command)
    total_memories = subprocess.getoutput(total_memories_command)
    available_memories_list = [int(item.split(':')[1].strip('MiB').strip('')) for item in available_memories.split('\n')]
    total_memories_list = [int(item.split(':')[1].strip('MiB').strip('')) for item in total_memories.split('\n')]
    return np.asarray(available_memories_list),np.asarray(total_memories_list)


def get_total_memories():
    #total_memories_command = 'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free'
    total_memories_command = 'nvidia-smi -q -d Memory | grep -A4 GPU | grep Total'
    #total_memories = subprocess.getoutput(total_memories_command)
    total_memories = subprocess.getoutput(total_memories_command)
    total_memories_list = [int(item.split(':')[1].strip('MiB').strip('')) for item in total_memories.split('\n')]
    return np.asarray(total_memories_list)


def get_available_memories():
    available_memories_command = 'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free'
    #total_memories_command = 'nvidia-smi -q -d Memory | grep -A4 GPU | grep Total'
    available_memories = subprocess.getoutput(available_memories_command)
    #total_memories = subprocess.getoutput(total_memories_command)
    available_memories_list = [int(item.split(':')[1].strip('MiB').strip('')) for item in available_memories.split('\n')]
    return np.asarray(available_memories_list)


def get_available_gpu_ids(memory=None):
    if memory is None:
        memory = 0
    available_memories_list = get_available_memories()
    return np.where(available_memories_list>=memory)


def get_optimal_gpu_id(memory=None):
    if memory is None:
        memory = 0
    available_memories_list,total_memories_list = get_memories()
    memory_differences = available_memories_list - memory
    queried_list = np.where(memory_differences >= 0, memory_differences, total_memories_list)
    gid = np.argmin(queried_list-memory)
    if available_memories_list[gid] < memory:
        gid = -1
    return gid


def get_optimal_gpu_id_with_memory_info(available_memories_list,total_memories_list,memory=None):
    if memory is None:
        memory = 0
    memory_differences = available_memories_list - memory
    queried_list = np.where(memory_differences >= 0, memory_differences, total_memories_list)
    gid = np.argmin(queried_list-memory)
    if available_memories_list[gid] < memory:
        gid = -1
    return gid
