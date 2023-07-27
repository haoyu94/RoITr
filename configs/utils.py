import yaml


def load_config(path):
    '''
    Load config file
    :param path: path to config file
    :return: a dictionary consisting of loaded configuration, sub dictionaries will be merged
    '''
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config
