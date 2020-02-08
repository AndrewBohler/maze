import numpy as np
import random

class Map:

    def __init__(self, x=10, y=10):
        self.x = x
        self.y = y
        self.start = None
        self.end = None
        self.map = np.zeros((x, y), dtype=int)

        self.gen()
        
    def gen(self, genstyle="default"):

        if genstyle == "default":
            print("Generating map...")

            # Initiate array
            self.map = np.random.randint(0, 1, (self.x, self.y), dtype=int)
            
            # Create borders
            for col in range(0, self.x):
                self.map[col, 0] = 1
                self.map[col, self.y-1] = 1

            for row in range(0, self.y):
                self.map[0, row] = 1
                self.map[self.x-1, row] = 1

            # Clear start and end positions
            self.start = np.random.randint(1, self.x-1)
            self.end = np.random.randint(1, self.x-1)

            self.map[self.start, 0] = 0
            self.map[self.end, self.y-1] = 0

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

            # create x and y walls by doing some matrix math...?

            # while xwalls:
            #     x = xwalls.pop()
            #     while ywalls:
            #         y = ywalls.pop()
            #         ref = self.map[x-1:x+1, y-1:y+1]



            # for x in range (1, self.x):
            #     if random.choice([True, False]):
            #         pos = np.random.randint(1, self.y-2, 2)
            #         print(pos)
            #         for y in range(pos[0], (pos[1]+1)):
            #             self.map[x, y] = 1

            # for y in range (1, self.x):
            #     if random.choice([True, False]):
            #         pos = np.random.randint(1, self.x-2, 2)
            #         print(pos)
            #         for x in range(pos[0], (pos[1]+1)):
            #             self.map[x, y] = 1


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
    
    map = Map(15, 15)

    print("\nThank you for using maze.py, have a nice day!\n")
    
            
