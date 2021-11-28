from datasets import load_dataset


def print_dict(d: dict, name: str = None):
    if name is not None:
        print(f'===== {name} =====')
    for k, v in d.items():
        print(f'{k} = {v}')
