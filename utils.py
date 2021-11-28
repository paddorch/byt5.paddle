from datasets import load_dataset


def print_dict(d: dict, name: str = None):
    if name is not None:
        s = f'===== {name} ====='
        print(s)

    for k, v in d.items():
        print(f'{k} = {v}')

    if name is not None:
        print(''.join(['=' for _ in range(len(s))]))
