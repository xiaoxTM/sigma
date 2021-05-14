import socketserver
from . import nvidia_gpus as ng
import logging
from multiprocessing import Lock,Process,Manager
import time
import hashlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s:%(asctime)s:%(module)s@%(lineno)s]:%(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

server = None

class Server():
    def __init__(self,includes=None,excludes=None,intervals=None):
        super(Server,self).__init__()
        self.manager = Manager()
        self.todo = self.manager.list()
        self.running = self.manager.list()
        self.total_memories = ng.get_total_memories()
        num_of_gpus = ng.get_num_of_gpus()
        if includes is None:
            includes = range(num_of_gpus)
        elif isinstance(includes,int):
            includes = [includes]
        self.includes = includes
        if excludes is not None:
            if isinstance(excludes,int):
                excludes = [excludes]
            self.includes = list(set(self.includes)-set(excludes))
        logger.info('GPUs available:{}'.format(self.includes))
        self.lock = Lock()
        self.intervals = intervals
        if self.intervals is None:
            self.intervals = 30
        run = Process(target=self.run,args=(self.todo,self.running))
        run.daemon = True
        run.start()

        observer = Process(target=self.observe,args=(self.running,self.intervals))
        observer.daemon = True
        observer.start()

    def observe(self,running,intervals):
        while True:
            cnt = time.time()
            self.lock.acquire()
            for idx, (start,gpu_id) in enumerate(running[::-1]):
                elapsed = cnt - start
                if elapsed >= intervals:
                    logger.info(f'GPU-{gpu_id} available again')
                    running.pop(idx)
            self.lock.release()

    def check_memory_capacity(self,task,max_memory):
        memory = task.get('memory',0)
        if memory is None:
            memory = 0
        assert memory <= max_memory

    def report(self,idlen=55):
        self.lock.acquire()
        message = ''
        for task in self.todo:
            if message == '':
                message = '{}=>{}'.format(task[0][:idlen],str(task[3]))
            else:
                message = '{}\n{}=>{}'.format(message,task[0][:idlen],str(task[3]))
        self.lock.release()
        return message

    def register(self,socket,address,task):
        self.check_memory_capacity(task,self.total_memories.max())
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        key = hashlib.sha224(bytes(timestamp,'utf-8')).hexdigest()
        self.lock.acquire()
        self.todo.append([key,socket,address,task])
        self.lock.release()

    def move(self,taskid,position):
        if position <0 or position > len(self.todo):
            return -1
        idlen = len(taskid)
        found_list = []
        self.lock.acquire()
        for idx,task in enumerate(self.todo):
            if task[0][:idlen] == taskid:
                found_list.append(idx)
        if len(found_list) == 1 and found_list[0] != position:
            self.todo[found_list[0]],self.todo[position] = self.todo[position],self.todo[found_list[0]]
        self.lock.release()
        return len(found_list)

    def delete(self,taskid,multiple=False):
        idlen = len(taskid)
        self.lock.acquire()
        num = 0
        if taskid == '*':
            for _,socket,address,_ in self.todo:
                socket.sendto(bytes('{"response":"CANCEL"}','utf-8'),address)
            num = len(self.todo)
            self.todo = self.manager.list()
        else:
            found_list = []
            reverse = False
            if taskid[0] == '~':
                taskid = taskid[1:]
                reverse = True
            for idx,task in enumerate(self.todo):
                if task[0][:idlen] == taskid:
                    found_list.append(idx)
            if not reverse:
                if len(found_list) >= 1 or multiple:
                    for idx in found_list[::-1]:
                        job = self.todo.pop(idx)
                        job[1].sendto(bytes('{"response":"CANCEL"}','utf-8'),job[2])
            else:
                found_list = list(set(range(self.todo))-set(found_list))
                for idx in found_list[::-1]:
                    job = self.todo.pop(idx)
                    job[1].sendto(bytes('{"response":"CANCEL"}','utf-8'),job[2])
            num = len(found_list)
        self.lock.release()
        return num

    def run(self,todo,running):
        while True:
            available_memories = ng.get_available_memories()
            self.lock.acquire()
            todo_pop_index = []
            for idx, (_,socket,address,task) in enumerate(todo):
                gpu_id = task.get('gpus',None)
                gid = -1
                if gpu_id is not None:
                    memory = task.get('memory',0)
                    if memory is None:
                        memory = 0
                    if available_memories[gpu_id] - memory >= 0:
                        gid = gpu_id
                else:
                    gid =ng.get_optimal_gpu_id_with_memory_info(available_memories,self.total_memories,task.get('memory',0))
                used_gpus = [gpu for (_,gpu) in running]
                if gid >=0 and gid not in used_gpus and gid in self.includes:
                    message = {"response":"OK","ans":gid}
                    logger.info('GPU-{} dispatched'.format(gid))
                    socket.sendto(bytes(str(message),'utf-8'),address)
                    todo_pop_index.append(idx)
                    running.append([time.time(),gid])
            for idx in todo_pop_index[::-1]:
                todo.pop(idx)
            self.lock.release()

class RequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        task = eval(self.request[0].strip())
        socket = self.request[1]
        logger.info("{}/{}".format(self.client_address[0],task['action']))
        action = task.pop('action')
        if action == 'register':
            gpu_id = task.get('gpus',None)
            if gpu_id is not None and gpu_id not in server.includes:
                socket.sendto(bytes('{"response":"ERROR","ans":"required GPU excluded by the server"}','utf-8'),self.client_address)
            server.register(socket,self.client_address,task)
        elif action == 'report':
            gpu_id = task.get('gpus')
        elif action == 'query':
            idlen = task['idlen']
            message = server.report(idlen)
            socket.sendto(bytes(message,'utf-8'),self.client_address)
        elif action == 'delete':
            key = task['id']
            multiple = task['multiple']
            num = server.delete(key,multiple)
            socket.sendto(bytes(str(num),'utf-8'),self.client_address)
        elif action == 'move':
            key = task['id']
            position = task['position']
            server.move(key,position)

# run the daemon
def run(args):
    server = Server(args.includes,args.excludes)
    logger.info('listening at {}:{}'.format(args.host,args.port))
    with socketserver.UDPServer((args.host,args.port),RequestHandler) as ser:
        ser.serve_forever()