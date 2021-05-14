import json
import os
import os.path
from . import nvidia_gpus as ng
import subprocess

def load_tasks(filename):
    if os.path.exists(filename):
        with open(filename,'r') as fp:
            tasks = json.load(fp)
    else:
        tasks = json.loads(filename)
    if isinstance(tasks,dict):
        tasks = [tasks]
    task_list = []
    total_memories = ng.get_total_memories()
    for task in tasks:
        gpu_id = task.get('gpus',None)
        memory = task.get('memory',None)
        if gpu_id is not None:
            if memory is not None:
                assert memory <= total_memories[gpu_id], f'GPU {gpu_id} has less memory than {memory}'
        elif memory is not None:
            assert memory <= total_memories.max(), f'memory {memory} is bigger than the max memory'
        task_list.append(Task(task['command'],task.get('block',True),memory,gpu_id,task.get('message','')))
    return task_list


class Task():
    def __init__(self,command,block=True,memory=None,gpus=None,message=None):
        self.command = command
        self.memory = memory
        self.message = message
        self.proc = None
        self.block = block
        self.gpus = gpus

    def run(self,gpu_id):
        assert self.gpus is not None or gpu_id != self.gpus, f'wrong gpu id[{gpu_id}] if gpus are specified by [gpus={self.gpus}] parameter'
        env = os.environ
        env.update({'CUDA_VISIBLE_DEVICES':str(gpu_id)})
        self.proc = subprocess.Popen(self.command,shell=True,stdin=subprocess.PIPE,env=env)
        if self.block:
            self.proc.wait()
