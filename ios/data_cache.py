from typing import Any

class DataCache():
    def __init__(self,capacity:int) -> None:
        assert capacity > 0
        self.capacity = capacity
        self.cache = []

    def __call__(self,index:int,*args:Any) -> Any:
        if index < self.capacity:
            self.cache.append(args)
        else:
            args = self.cache[index%self.capacity]
        return args
