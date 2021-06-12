import yaml

def merge_with_args(config,args):
    args = vars(args)
    print(args)
    for k,v in args.items():
        if k in config.keys():
            if v is not None:
                if config[k] is not None and type(config[k]) != type(v):
                    raise TypeError('config `{}` is type `{}`, while args is `{}`'.format(k,type(config[k]),type(v)))
                config.update({k:v})
        else:
            config.update({k:v})
    return config

def load(filename):
    with open(filename,'r') as f:
        configs = yaml.safe_load(f)
    return configs