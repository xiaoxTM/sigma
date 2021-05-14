from . import server
from . import client
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--port',type=int,default='10930')
parser.add_argument('--host',type=str,default='localhost')

subparsers = parser.add_subparsers()

# run daemon
parser_run = subparsers.add_parser('run')
parser_run.set_defaults(callback=lambda args:server.run(args))
parser_run.add_argument('--host',type=str,default='localhost')
parser_run.add_argument('--port',type=int,default=10930)
parser_run.add_argument('--intervals',type=float,default=30)
parser_run.add_argument('--includes',type=int,nargs='+',default=None)
parser_run.add_argument('--excludes',type=int,nargs='+',default=None)

# register
parser_register = subparsers.add_parser('register')
parser_register.set_defaults(callback=lambda args:client.register(args))
parser_register.add_argument('--config',type=str,required=True)

# query
parser_query = subparsers.add_parser('query')
parser_query.set_defaults(callback=lambda args:client.query(args))
parser_query.add_argument('--id-len',type=int,default=8)

# delete
parser_delete = subparsers.add_parser('delete')
parser_delete.set_defaults(callback=lambda args:client.delete(args))
parser_delete.add_argument('--id',type=str,required=True)
parser_delete.add_argument('--id-len',type=int,default=16)
parser_delete.add_argument('--delete-multiple',action='store_true',default=False)

# move
parser_move = subparsers.add_parser('move')
parser_move.set_defaults(callback=lambda args:client.move(args))
parser_move.add_argument('--id',type=str,required=True)
parser_move.add_argument('--position',type=int,required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    args.callback(args)

