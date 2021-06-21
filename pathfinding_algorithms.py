from maze import Maze
from maze import NORTH, SOUTH, EAST, WEST
import numpy as np
from typing import *
from typing_extensions import TypeAlias

Path: TypeAlias = Tuple[Tuple[int, int]]

def depth_first_search(maze: Maze) -> Path:
    Node = NamedTuple(
        "Node", (
            ("position", Tuple[int, int]),
            ("edges", List[Tuple[int, int]])
        )
    )

    dirs = NORTH, SOUTH, EAST, WEST # one-hot 4-bit numbers
    edges = (0, 1), (0, -1), (1, 0), (-1, 0)

    def _get_edges(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        "unpacks edges from single value in maze's ndarray"
        bit_pack = maze.data[pos]
        return [e for d, e in zip(dirs, edges) if bit_pack & d]

    start = Node(maze.start, _get_edges(maze.start))

    stack = [start]
    seen = np.zeros(maze.data.shape, dtype=bool)

    while stack:
        node = stack[-1]

        print(node.position)

        if node.position == maze.end:
            break

        elif not node.edges:
            seen[node.position] = False
            stack.pop()
        
        else:
            edge = node.edges.pop()
            pos = node.position[0] + edge[0], node.position[1] + edge[1]
            
            if not seen[pos]:
                seen[pos] = True
                stack.append(Node(pos, _get_edges(pos)))

    return tuple(node.position for node in stack)


if __name__ == "__main__":
    maze = Maze(10, 10)
    maze.print()
    path = depth_first_search(maze)

    print(*path, sep="\n")
    print("didn't fail")


        
