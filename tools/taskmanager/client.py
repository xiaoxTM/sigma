import socket
import sys
import argparse
from . import tasks
import logging
from sigma.fontstyles import colors

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s:%(asctime)s:%(module)s@%(lineno)s]:%(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

# register
def register(args):
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    task_list = tasks.load_tasks(args.config)
    for task in task_list:
        logger.info('acquiring GPU resource')
        message = str({'action':'register','command':task.command,'memory':task.memory,'gpus':task.gpus})
        sock.sendto(bytes(message,'utf-8'),(args.host,args.port))
        received = eval(str(sock.recv(4096),'utf-8').strip())
        response = received.get('response',None)
        if response == 'OK':
            gpu_id = received.get('ans')
            logger.info('run job {}@GPU-{}'.format(colors.green(task.command),colors.blue(str(gpu_id))))
            task.run(gpu_id)
            message = str({'action':'report','gpus':gpu_id})
            sock.sendto(bytes(message,'utf-8'),(args.host,args.port))
        elif response == 'ERROR':
            logger.error(colors.red('ERROR:'),received.get('ans','register failed'))
        elif response == 'CANCEL':
            logger.info('job {} cancelled'.format(colors.yellow(task.command)))
        else:
            logger.warning('unknown response')

# query
def query(args):
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    message = str({"action":"query","idlen":args.id_len})
    sock.sendto(bytes(message,'utf-8'),(args.host,args.port))
    received = str(sock.recv(4096),'utf-8').strip()
    if received == '':
        received = 'no job in queue'
    else:
        received = 'jobs in queue:\n{}'.format(received)
    logger.info(received)

# delete
def delete(args):
    assert args.id is not None
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    message = str({"action":"delete","multiple":args.delete_multiple,'id':args.id})
    sock.sendto(bytes(message,'utf-8'),(args.host,args.port))
    received = str(sock.recv(4096),'utf-8').strip()

# move
def move(args):
    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    assert args.id is not None
    assert args.position is not None
    message = str({"action":"move","id":args.id,"position":args.position})
    sock.sendto(bytes(message,'utf-8'),(args.host,args.port))
