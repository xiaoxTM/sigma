from . import nvidia_gpus as ng
import subprocess
import json
import os

class Task():
    def __init__(self,command,block=False,memory=None,gpus=None,message=None):
        self.command = command
        self.memory = memory
        self.returncode = 0
        self.isdone = False
        self.message = message
        self.proc = None
        self.block = block
        self.gpus = gpus

    def run(self,gpu_id):
        assert self.gpus is not None or gpu_id != self.gpus, f'wrong gpu id[{gpu_id}] if gpus are specified by [gpus={self.gpus}] parameter'
        env = os.environ
        env.update({'CUDA_VISIBLE_DEVICES':str(gpu_id)})
        #print('env:',env)
        self.proc = subprocess.Popen(self.command,shell=True,stdin=subprocess.PIPE,env=env)
        #self.proc = subprocess.Popen(self.command.format(gpu_id),shell=True,stdin=subprocess.PIPE)
        if self.block:
            self.proc.wait()

    def check(self):
        self.returncode = self.proc.poll()

    @property
    def done(self):
        self.returncode = self.proc.poll()
        if self.returncode is None:
            return False
        else:
            return True


class TaskManager():
    def __init__(self,config_filename_or_string,poolsize=None,includes=None,excludes=None):
        # config_filename: path to json configure file or str
        # poolsize: int
        # includes: int or List of int
        # excludes: int or List of int, higher prior than includes
        self.todo = []
        self.running = []
        self.done = []
        self.total_memories = ng.get_total_memories()
        self.load_config(config_filename_or_string)
        if poolsize is None:
            poolsize = len(self.todo)
        self.poolsize = poolsize
        num_of_gpus = ng.get_num_of_gpus()
        if includes is None:
            includes = range(num_of_gpus)
        elif isinstance(includes,int):
            includes = [includes]
        self.includes = includes
        if excludes is not None and isinstance(excludes,int):
            excludes = [excludes]
        if excludes is not None:
            self.includes = list(set(self.includes)-set(excludes))

    def load_config(self,filename):
        if os.path.exists(filename):
            with open(filename,'r') as fp:
                tasks = json.load(fp)
        else:
            tasks = json.loads(filename)
        if isinstance(tasks,dict):
            tasks = [tasks]
        for task in tasks:
            gpu_id = task.get('gpus',None)
            memory = task.get('memory',None)
            if gpu_id is not None:
                if memory is not None:
                    assert memory <= self.total_memories[gpu_id], f'GPU {gpu_id} has less memory than {memory}'
            elif memory is not None:
                assert memory <= self.total_memories.max(), f'memory {memory} is bigger than the max memory'
            self.todo.append(Task(task['command'],task.get('block',False),memory,gpu_id,task.get('message','')))

    def check_memory_capacity(self,max_memory):
        for i,todo in enumerate(self.todo):
            if todo.memory is not None and todo.memory > max_memory:
                raise ValueError(f'warning: task {i} requires memory that is big than the max memory that GPU can provide')

    def run(self):
        self.check_memory_capacity(self.total_memories.max())
        while True:
           if len(self.todo) == 0 and len(self.running) == 0:
               exit()
           idx = 0
           # check done task, add to done list and remove from running list
           while idx < len(self.running):
               task = self.running[idx]
               if task.done:
                   task.proc.terminate()
                   self.done.append(task)
                   self.running.pop(idx)
               else:
                   idx += 1
           if len(self.running) < self.poolsize:
               available_memories = ng.get_available_memories()
               for idx,task in enumerate(self.todo):
                   gid = ng.get_optimal_gpu_id_with_memory_info(available_memories,self.total_memories,task.memory)
                   if gid >= 0 and gid in self.includes:
                       task.run(gid)
                       self.running.append(task)
                       self.todo.pop(idx)
                       break
