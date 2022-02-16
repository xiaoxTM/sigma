import yaml
from types import SimpleNamespace
import json

def merge_args(config,args):
    args = vars(args)
    for k,v in args.items():
        if k in config.keys():
            if v is not None:
                if config[k] is not None and type(config[k]) != type(v):
                    raise TypeError('config `{}` is type `{}`, while args is `{}`'.format(k,type(config[k]),type(v)))
                config.update({k:v})
        else:
            config.update({k:v})
    return config


def load(filename, *args, **kwargs):
    with open(filename,'r') as f:
        configs = yaml.load(f, *args, **kwargs)
    return configs


def namespace(d):
    # Sample
    return json.loads(json.dumps(d), object_hook=lambda item: SimpleNamespace(**item))


def save(filename, config):
    with open(filename,'w') as f:
        yaml.dump(vars(config),f)
