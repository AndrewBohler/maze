import numpy as np
import random

class Map:

    def __init__(self, x=10, y=10, *args, **kwargs):
        self.x = x
        self.y = y
        self.start = None
        self.end = None
        self.map = np.zeros((x, y), dtype=int)

        self.gen()
        
    def gen(self, genstyle="default", *args, **kwargs):

        print("Generating map...")
        
        # Create borders
        self.map[0, :] = 1
        self.map[-1, :] = 1
        self.map[:, 0] = 1
        self.map[:, -1] = 1

        # Clear start and end positions
        self.start = (random.randint(1, self.x-1), 0)
        self.end = (random.randint(1, self.x-1), -1)

        self.map[self.start] = 0
        self.map[self.end] = 0

        if genstyle == "default":
            print('generation type = "default"')

            self.map[1:-1, 1:-1] = 1

            # Keep track of covered ground using exploration matrix
            explored = np.ones((self.x, self.y), dtype = int)

            # pre-explore map border (still unexplored while > 0)
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
                self.map[x, y] = 0

                # Right, left, up, down
                direction = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                random.shuffle(direction)

                while direction:
                    dx, dy = direction.pop()
                    if not 0 < dx < self.x:
                        print(f"[error] pos {dx}, {dy} is outside of map {self.x}, {self.y}")
                    elif not 0 < dy < self.y:
                        print(f"[error] pos {dx}, {dy} is outside of map {self.x}, {self.y}")

                    # if unexplored then move
                    elif explored[dx, dy] > 0:
                        explored[x-1:x+1, y-1:y+1] = 0
                        x, y = dx, dy
                        break

                    # chance to break (explored) walls
                    else:
                        chance = random.randint(1, 100)
                        if chance > 99: # clear explored chance
                            explored[x-1:x+1, y-1:y+1] = 0
                            x, y = dx, dy
                            break

                    # reposition if all directions are already explored
                    if not direction:
                        for i in range(1, self.x):
                            for j in range(1, self.y):
                                if explored[i, j] > 0:
                                    x, y = i, j
                                    self.map[x, y] = 0

                        

        elif genstyle == "random tiles":
            print('generation type = "random tiles"')
            walls = np.random.randint(0, 2, (self.x-2, self.y-2), dtype=int)
            self.map[1:-1, 1:-1] = walls

        elif genstyle == "random walls":
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
                            self.map[x+i, y] = 1
                        if x - i > 0:
                            self.map[x-i, y] = 1
                            
                elif direction == 2:
                    for i in range(wall_size):
                        if y + i < self.y:
                            self.map[x, y+i] = 1
                        if y - i > 0:
                            self.map[x, y-i] = 1

        else:
            print("[Error] generation type unknown")

        print("Map generation complete:\n")
        self.display()

    def display(self):
        # Print the map to the terminal

        # formatting
        for y in range(self.y):
            row = []
            line = " "
            for x in range(self.x):
                if self.map[y, x] == 0:
                    row.append(" ")
                elif self.map[y, x] == 1:
                    row.append("#")
                else:
                    row.append("E")
            print("  " + line.join(row))


if __name__ == "__main__":
    print("maze.py is now running, this is a WIP\n")
    
    map = Map(15, 15, "random walls")

    print("\nThank you for using maze.py, have a nice day!\n")
    
            
