import numpy as np
import random
import time

class Maze:

    def __init__(self, x=10, y=10, *args, **kwargs):
        self.x = x
        self.y = y
        self.genstyle = kwargs.get("genstyle", "default")
        self.start = None
        self.end = None
        self.tiles = np.zeros((x, y), dtype=int)

        self._generate()
    
    def regenerate(self, *args, **kwargs):
        self.tiles[:, :] = 0
        self._generate()

    def _generate(self, *args, **kwargs):
        self.genstyle = kwargs.get("genstyle", self.genstyle)

        print("Generating maze...")
        
        # Create borders
        self.tiles[0, :] = 1
        self.tiles[-1, :] = 1
        self.tiles[:, 0] = 1
        self.tiles[:, -1] = 1

        # Clear start and end positions
        self.start = (random.randint(1, self.x-1), 0)
        self.end = (random.randint(1, self.x-1), self.y-1)

        self.tiles[self.start] = 0
        self.tiles[self.end] = 0

        if self.genstyle == "default":
            print('generation type = "default"')

            self.tiles[1:-1, 1:-1] = 1

            # Keep track of covered ground using exploration matrix
            explored = np.ones((self.x, self.y), dtype = int)

            # pre-explore mmaze border (still unexplored while > 0)
            explored[0, :] = 0
            explored[-1, :] = 0
            explored[:, 0] = 0
            explored[:, -1] = 0

            # Starting position 1 tile into maze from entrance
            x, y = self.start[0], self.start[1]
            if x == 0: x += 1
            elif x == self.x: x -= 1
            if y == 0: y += 1
            elif y == self.y: y -= 1

            while explored.any() > 0:
                explored[x, y] = 0
                self.tiles[x, y] = 0

                # Right, left, up, down
                direction = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                random.shuffle(direction)

                while direction:
                    random.shuffle(direction)
                    dx, dy = direction.pop()
                    if not 0 < dx < self.x:
                        # print(f"[error] pos {dx}, {dy} is outside of maze {self.x}, {self.y}")
                        pass
                    elif not 0 < dy < self.y:
                        # print(f"[error] pos {dx}, {dy} is outside of maze {self.x}, {self.y}")
                        pass

                    # if unexplored then move
                    elif explored[dx, dy] > 0:
                        explored[x-1:x+1, y-1:y+1] = 0
                        x, y = dx, dy
                        break

                    # reposition if all directions are already explored
                    if not direction:
                        for i in range(1, self.x):
                            for j in range(1, self.y):
                                if explored[i, j] > 0:
                                    x, y = i, j
                                    self.tiles[x, y] = 0
                                    break

        elif self.genstyle == "random tiles":
            print('generation type = "random tiles"')
            walls = np.random.randint(0, 2, (self.x-2, self.y-2), dtype=int)
            self.tiles[1:-1, 1:-1] = walls

        elif self.genstyle == "random walls":
            print('generation type = "random walls"')

            # Create some walls
            xwalls = list(range(1, self.x))
            ywalls = list(range(1, self.y))

            random.shuffle(xwalls)
            random.shuffle(ywalls)

            wall_count = 10

            while wall_count > 0:
                wall_count -= 1
                x = random.choice(xwalls)
                y = random.choice(ywalls)
                direction = random.randint(1, 2)
                wall_size = random.randint(1, 5)

                if direction == 1:
                    for i in range(wall_size):
                        if x + i < self.x:
                            self.tiles[x+i, y] = 1
                        if x - i > 0:
                            self.tiles[x-i, y] = 1
                            
                elif direction == 2:
                    for i in range(wall_size):
                        if y + i < self.y:
                            self.tiles[x, y+i] = 1
                        if y - i > 0:
                            self.tiles[x, y-i] = 1

        else:
            print("[Error] generation type unknown")

        print("Maze generation complete:\n")

    def display(self):
        # Print the maze to the terminal

        # formatting
        for x in range(self.x):
            row = []
            line = " "
            for y in range(self.y):
                if self.tiles[x, y] == 0:
                    row.append(" ")
                elif self.tiles[x, y] == 1:
                    row.append("#")
                else:
                    row.append("E")
            print("  " + line.join(row))

class PathFinder:
    def __init__(self, maze=None):
        self._maze = maze
        self.pos = [0, 0]
        self._data = {}

    def assign_maze(self, maze: Maze):
        """Assign a new maze AND clear data"""
        self._maze = maze
        self._data = {}

    def solve(self, *args, **kwargs):
        def _setup_solve_data():
            self._data["path"] = []

        def _validate_maze():
            if not isinstance(self._maze, Maze):
                print(f"[error] {self._maze} is not a {Maze}")
                return False

        def _check_unwalked(x, y, path_number):
            if (x, y) in self._data["path"][path_number]: return False
            else: return True

        # start pathing the maze
        path_number = 0
        x, y = self._maze.start
        path = [(x, y)]

        # do some recursive funciton calls to calculate path
        # if end is found save path to self._data["path"][path_number]
        # if shortest_path == True increment path_number, calculate next path
        # and repeat until all possible paths tried
        # output results


    def create_nodes(self):
        """ Map the maze into a list of nodes"""
        self._data["nodes"] = [
            [
                [] if self._is_node((x, y)) else None for y in range(self._maze.y)
            ] for x in range(self._maze.x)
        ]

        for x in range(self._maze.x):
            for y in range(self._maze.y):
                if not self._data["nodes"][x][y] is None:
                    self._connect_nodes((x, y))


    def _connect_nodes(self, node: tuple):
        """connect nodes together"""
        x, y = node
        
        # Left
        dx = x
        while dx > 0 and self._maze.tiles[dx, y] == 0:
            dx -= 1
            if self._is_node((dx, y)):
                self._data["nodes"][x][y].append((dx, y))

        # Right
        dx = x
        while dx < self._maze.x-1 and self._maze.tiles[dx, y] == 0:
            dx += 1
            if self._is_node((dx, y)):
                self._data["nodes"][x][y].append((dx, y))
        # Down
        dy = y
        while dy > 0 and self._maze.tiles[x, dy] == 0:
            dy -= 1
            if self._is_node((x, dy)):
                self._data["nodes"][x][y].append((x, dy))

        # Up
        dy = y
        while dy < self._maze.y-1 and self._maze.tiles[x, dy] == 0:
            dy += 1
            if self._is_node((x, dy)):
                self._data["nodes"][x][y].append((x, dy))

    def _is_node(self, coords: tuple) -> bool:
        # The start and end of a maze are nodes automatically
        if coords == self._maze.start or coords == self._maze.end:
            return True

        x, y = coords

        # If all 8 surrounding tiles are clear then ignore
        if not self._maze.tiles[x-1:x+2, y-1:y+2].any() > 0:
            return False

        # Check if the cardinal directions are open
        left, right, up, down = False, False, False, False
        if x > 0 and self._maze.tiles[x-1, y] == 0:
            left = True
        if x < self._maze.x-1 and self._maze.tiles[x+1, y] == 0:
            right = True
        if y > 0 and self._maze.tiles[x, y-1] == 0:
            down = True
        if y < self._maze.y-1 and self._maze.tiles[x, y+1] == 0:
            up = True

        # straight paths and deadends are not nodes, but corners are!
        for xbool in [left, right]:
            for ybool in [up, down]:
                if xbool and ybool:
                    return True

        return False

    def display_nodes(self):
        if not 'nodes' in self._data:
            print("[error] cannont display nodes, data missing")
            return
        for x in range(self._maze.x):
            line = [' ']
            for y in range(self._maze.y):
                if self._maze.tiles[x, y] == 1:
                    line.append('#')
                elif self._maze.tiles[x, y] == 0 \
                    and self._data['nodes'][x][y] != None:
                    line.append('.')
                elif self._maze.tiles[x, y] == 0:
                    line.append(' ')
                else:
                    line.append('e')
            print(' '.join(line))

    def show_path(self):
        pass


if __name__ == "__main__":
    print("maze.py is now running, this is a WIP\n")
    
    test = Maze(20, 60)
    test.display()

    guy = PathFinder(test)
    guy.create_nodes()

    print('\nwith nodes ".":\n')
    guy.display_nodes()

    time.sleep(5)
    test.regenerate(genstyle="random walls")
    guy.create_nodes()
    guy.display_nodes()

    time.sleep(5)
    test.regenerate(genstyle="random tiles")
    guy.create_nodes()
    guy.display_nodes()

    print("\nThank you for using maze.py, have a nice day!\n")         
