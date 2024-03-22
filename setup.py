import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname: str) -> str:
    """Reads README file
    Args:
        fname (str): path to readme file
    Returns:
        str: contents in readme
    """
    full_path = os.path.join(os.path.dirname(__file__), fname)
    with open(full_path, encoding="utf-8") as file:
        return file.read()


setup(
      name="TensorFlow_v_Pytorch",
      version="0.0.1",
      author="Marcus Allen",
      author_email="marcusCallen24@gmail.com",
      url="https://github.com/marcus-24/TensorFlow_v_Pytorch",
      long_description=read('README.md'),
      py_modules=[]
)