import os

def ensure_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def flatten(l):
    return [item for sublist in l for item in sublist]
