import colorama
import random
import time
from typing import *

import numpy as np

colorama.init()

# one-hot representation
NORTH = int("0001", 2)
EAST = int("0010", 2)
SOUTH = int("0100", 2)
WEST = int("1000", 2)

SYMBOLS = ""
# 4 BIT | west | south | east | north

# 00xx
SYMBOLS += " ∙∙└"
# 01xx
SYMBOLS += "∙│┌├" 
# 10xx
SYMBOLS += "∙┘─┴"
# 11xx
SYMBOLS += "┐┤┬┼"

def vec2(x=0, y=0, dtype=float) -> np.ndarray:
    return np.array((x, y), dtype=dtype)

def ivec2(x=0, y=0, dtype=int) -> np.ndarray:
    return np.array((x, y), dtype=dtype)

class Maze:

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

        self.start = random.randint(0, rows - 1), random.randint(0, cols - 1)
        self.end = random.randint(0, rows - 1), random.randint(0, cols - 1)

        # only first 4 bits indicating 4 edges along cardinal directions
        self.data = np.zeros((rows, cols), dtype=np.int8)

        self.generate()

        sf = np.vectorize(lambda x: SYMBOLS[x])
        self.symbols = sf(self.data)

    def generate(self):
        # tuples index ndarrays, using another array would act as a slice
        offsets = (1, 0), (0, 1), (-1, 0), (0, -1)
        edges_out = NORTH, EAST, SOUTH, WEST # this position -> next position
        edges_in = SOUTH, WEST, NORTH, EAST # next position -> this position
        
        start = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
        stack = [start]
        visited = np.zeros((self.rows, self.cols), dtype=bool)
        visited[start] = True

        # depth-first search, carving out path
        while stack:
            position = stack[-1]
            adjacent_unvisited = []

            for i in range(4):
                pos = position[0] + offsets[i][0], position[1] + offsets[i][1]
                # validate in bounds and unvisited
                if (0 <= pos[0] < self.rows
                    and 0 <= pos[1] < self.cols
                    and not visited[pos]
                ):
                    adjacent_unvisited.append((pos, edges_out[i], edges_in[i]))

            # if nowhere to go, backup to find unvisited area
            if not adjacent_unvisited:
                stack.pop()
                continue

            next_pos, edge_out, edge_in = random.choice(adjacent_unvisited)

            stack.append(next_pos)

            self.data[position] |= edge_out
            self.data[next_pos] |= edge_in
            visited[next_pos] = True

    
    def print(self):
        "print maze to terminal"

        print("start:", self.start)
        print("end:", self.end)

        # reversed rows because we're printing downwards
        print(*("".join(row) for row in reversed(self.symbols)), sep="\n")

if __name__ == "__main__":
    for i in range(10):
        print("iteration", i)
        maze = Maze(30, 30)
        # maze.print()

        # origin (0, 0) is bottom-left
        top_row = maze.data[-1]
        bottom_row = maze.data[0]
        left_col = maze.data.T[0]
        right_col = maze.data.T[-1]


        print("top row   : ", *(f"{colorama.Fore.RED if v & NORTH else colorama.Fore.GREEN}{bin(v + 16)[3:]}{colorama.Fore.RESET}" for v in top_row))
        print("bot row   : ", *(f"{colorama.Fore.RED if v & SOUTH else colorama.Fore.GREEN}{bin(v + 16)[3:]}{colorama.Fore.RESET}" for v in bottom_row))
        print("left_col  : ", *(f"{colorama.Fore.RED if v & WEST else colorama.Fore.GREEN}{bin(v + 16)[3:]}{colorama.Fore.RESET}" for v in left_col))
        print("right_col : ", *(f"{colorama.Fore.RED if v & EAST else colorama.Fore.GREEN}{bin(v + 16)[3:]}{colorama.Fore.RESET}" for v in right_col))

    print("I am finished")