import gpu_scheduler as gs

tm = gs.TaskManager('test.json',poolsize=2,excludes=3)
tm.run()
print('task manager splitter-----------------------------------------------------------')
tm2 = gs.TaskManager('{"command":"python3 main.py --gputest","block":true,"gpus":5,"memory":1000}',poolsize=2)
tm2.run()
