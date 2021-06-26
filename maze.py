import colorama
from colorama import Back, Fore, Style
from functools import cached_property
import random
import time
from typing import *

import numpy as np

colorama.init()

# one-hot representation
NORTH = int("0001", 2)
EAST  = int("0010", 2)
SOUTH = int("0100", 2)
WEST  = int("1000", 2)

# DEADEND = "∙"
DEADEND = " "

# 4 BIT | west | south | east | north
SYMBOLS = [" "] * 16

# 4 deadends are the same symbol
SYMBOLS[NORTH] = DEADEND
SYMBOLS[SOUTH] = DEADEND
SYMBOLS[EAST] = DEADEND
SYMBOLS[WEST] = DEADEND

SYMBOLS[NORTH | SOUTH] = "│"
SYMBOLS[EAST  | WEST]  = "─"

SYMBOLS[NORTH | EAST]  = "└"
SYMBOLS[EAST  | SOUTH] = "┌"
SYMBOLS[SOUTH | WEST]  = "┐"
SYMBOLS[WEST  | NORTH] = "┘"

SYMBOLS[WEST  | NORTH | EAST]  = "┴"
SYMBOLS[NORTH | EAST  | SOUTH] = "├"
SYMBOLS[EAST  | SOUTH | WEST]  = "┬"
SYMBOLS[SOUTH | WEST  | NORTH] = "┤"

SYMBOLS[NORTH | SOUTH | EAST | WEST] = "┼"

def vec2(x=0, y=0, dtype=float) -> np.ndarray:
    return np.array((x, y), dtype=dtype)

def ivec2(x=0, y=0, dtype=int) -> np.ndarray:
    return np.array((x, y), dtype=dtype)

class Maze:

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols

        self.start = random.randint(0, rows - 1), random.randint(0, cols - 1)
        self.end = random.randint(0, rows - 1), random.randint(0, cols - 1)

        # only first 4 bits indicating 4 edges along cardinal directions
        self.data = np.zeros((rows, cols), dtype=np.int8)

    @cached_property
    def symbols(self):
        sv = np.vectorize(lambda x: SYMBOLS[x])
        arr = sv(self.data)
        arr[self.start] = "S"
        arr[self.end] = "E"
        return arr

    def generate(self):
        # generating a new maze invalidates the symbols array, del the cache
        if hasattr(self, "symbols"):
            del self.symbols

        self.data[:, :] = 0

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
    
    def show_in_terminal(
        self,
        path: Optional[Tuple[Tuple[int, int]]]=None,
    ):
        "print maze to terminal"

        print("start:", self.start)
        print("end:", self.end)

        # reversed rows because we're printing downwards
        if path:
            path = set(path)
            color0 = Fore.BLACK  + Back.LIGHTBLACK_EX # not path
            color1 = Fore.YELLOW + Back.LIGHTBLACK_EX # path
            color2 = Fore.GREEN  + Back.LIGHTBLACK_EX # start
            color3 = Fore.RED    + Back.LIGHTBLACK_EX # end

            output = []
            for row in range(self.rows - 1, -1, -1):
                for col in range(self.cols):
                    symbol = self.symbols[row, col]

                    if (row, col) in path:
                        if (row, col) == self.start:
                            symbol = f"{color2}{symbol}"

                        elif (row, col) == self.end:
                            symbol = f"{color3}{symbol}"

                        else:
                            symbol = f"{color1}{symbol}"
                    else:
                        symbol = f"{color0}{symbol}"

                    output.append(symbol)
                output.append(Style.RESET_ALL)
                output.append("\n")
                
            print("".join(output), end=" ", flush=True)

        else:
            print(*("".join(row) for row in reversed(self.symbols)), sep="\n")

if __name__ == "__main__":
    for i in range(1):
        print("iteration", i)
        maze = Maze(30, 30)
        maze.generate()
        maze.show_in_terminal()

    print("I am finished")
