import pathlib
import os

def compute_root_dir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return pathlib.Path(root_dir)


proj_root_dir = compute_root_dir()
default_image_path = str(proj_root_dir / 'imgs/1.jpg')