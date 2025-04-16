
from os.path import dirname, realpath
import sys
sys.path.append(dirname(dirname(realpath(__file__))))

import pickle
import os

current_directory=os.getcwd()
print(current_directory)
print()
results_directory = os.path.join(current_directory, "results")
def read(path=os.path.join(results_directory, "test_stats")):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    print(objects)

read()
