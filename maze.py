import copy
import math
from multiprocessing import Process
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import random
import sys
import time


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


class Maze:

    def __init__(self, x=10, y=10, *args, **kwargs):
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.num_tiles = x * y
        self.genstyle = kwargs.get("genstyle", "default")
        self.start = None
        self.end = None
        self.tiles = np.zeros((x, y), dtype=int)

        self._generate(**kwargs)
    
    def regenerate(self, *args, **kwargs):
        self.tiles.fill(0)
        self._generate(*args, **kwargs)

    def _generate(self, *args, **kwargs):
        self.genstyle = kwargs.get("genstyle", self.genstyle)

        print("Generating maze...")
        
        # Create borders
        self.tiles[0, :] = 1
        self.tiles[-1, :] = 1
        self.tiles[:, 0] = 1
        self.tiles[:, -1] = 1

        # Clear start and end positions
        self.start = (random.randint(1, self.x-2), 0)
        self.end = (random.randint(1, self.x-2), self.y-1)

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
            
            x, y = self.start
            # Starting position 1 tile into maze from entrance
            if x == 0: x += 1
            elif x == self.x: x -= 1
            if y == 0: y += 1
            elif y == self.y: y -= 1

            def _get_unexplored_coords(self) -> tuple:
                """returns coords of unexplored tile, or None"""
                for i in range(1, self.x):
                    for j in range(1, self.y):
                        if explored[i, j] > 0:
                            yield (i, j)
                yield (None, None)

            while explored.any() > 0:
                explored[x, y] = 0
                self.tiles[x, y] = 0
                find_unexplored = _get_unexplored_coords(self)

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
                        x, y = next(find_unexplored)
                        if x == None or y == None:
                            pass
                        else:
                            self.tiles[x, y] = 0
            
                explored_count = self.num_tiles - np.sum(explored)
                printProgressBar(
                    explored_count, self.num_tiles, f'Generating...',
                    f'{explored_count}/{self.num_tiles}', length=50,
                )

        elif self.genstyle == "random tiles":
            print('generation type = "random tiles"')
            walls = np.random.randint(0, 2, (self.x-2, self.y-2), dtype=int)
            self.tiles[1:-1, 1:-1] = walls

        elif self.genstyle == "random walls":
            print('generation type = "random walls"')

            wall_count = kwargs.get("wall_count", (self.x+self.y)//2)
            max_wall_size = kwargs.get("max_wall_size", (min([self.y, self.y])//4))

            while wall_count > 0:
                wall_count -= 1
                x = random.randint(1, self.x-1)
                y = random.randint(1, self.y-1)
                direction = random.randint(1, 2)
                wall_size = random.randint(1, max_wall_size)

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

        if kwargs.get('fill', False):
            """Narrow pathways by filling in elligible open space"""
            
            print('Filling in open space!!!!')

            def get_slice(p:tuple) -> np.array:
                return self.tiles[p[0]-1:p[0]+2, p[1]-1:p[1]+2]

            def no_direction(a: np.array) -> bool:
                if a[1, 0] == 0 and a[1, 2] == 0 and \
                    a[0, 1] == 0 and a[2, 1] == 0:
                    return True
                else: return False

            def double_wide(a: np.array) -> bool:
                up = np.array([[1, 1, 1], [0, 0 , 0], [0, 0, 0]])
                down = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
                left = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
                right = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
                for d in [up, down, left, right]:
                    if np.array_equal(a, d): return True
                return False

            total_iterations = (self.x-2)*(self.y-2)*2
            i = 0

            # check "double wide"
            for x in range(1, self.x-1):
                for y in range(1, self.y-1):
                    i += 1
                    if self.tiles[x, y] == 0:
                        a = get_slice((x, y))
                        if double_wide(a):
                            self.tiles[x, y] = 1

                    printProgressBar(
                        i, total_iterations, length=50,
                        suffix=(f'{i}/{total_iterations}'))

            # check "no direction"
            for x in range(1, self.x-1):
                for y in range(1, self.y-1):
                    i += 1
                    if self.tiles[x, y] == 0:
                        a = get_slice((x, y))
                        if no_direction(a):
                            self.tiles[x, y] = 1

                    printProgressBar(
                        i, total_iterations, length=50,
                        suffix=(f'{i}/{total_iterations}'))


        print("Maze generation complete:\n")

    def display(self):
        """Print the maze to the terminal"""

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
    def __init__(self,
    maze: Maze = None,
    memory_manager: SharedMemoryManager = None
    ):
        self._maze = maze
        self.display = None
        self.memory_manager = memory_manager
        self.pos = [0, 0]
        self._data = {}

        if self.memory_manager is None:
            self.memory_manager = SharedMemoryManager()
            self.memory_manager.start()

    def assign_maze(self, maze: Maze):
        """Assign a new maze AND clear data"""
        self._maze = maze
        # self._data = {}

    def solve(self, *args, **kwargs):
        start_time = time.time()
        time.sleep(0.000001) # prevent division by zero error

        def _setup_solve_data(**kwargs):
            
            self._data["node_depth"] = 0
            self._data["total_nodes"] = sum([len(x) for x in self._data["nodes"]])
            self._data["node_factorial"] = math.factorial(len(self._data["nodes"]))
            self._data["step_count"] = 0

            if not self._data.get('path_mem', False):
                print('[info] setting up shared memory')
                path_len = len(self._maze.tiles.flat)

                sys.setrecursionlimit(len(self._maze.tiles.flat)*2)

                self._data["path_mem"] = self.memory_manager.SharedMemory(self._maze.tiles.nbytes*2)
                self._data["path"] = np.ndarray(
                    (path_len, 2), dtype=int, buffer=self._data["path_mem"].buf)
                
                self._data["maze_mem"] = self.memory_manager.SharedMemory(
                    self._maze.tiles.nbytes)

                self._data['maze'] = np.ndarray(
                    self._maze.tiles.shape,
                    dtype=self._maze.tiles.dtype,
                    buffer=self._data['maze_mem'].buf
                )

                # state = running, step_count, step_per_sec
                self._data['display_state_mem'] = self.memory_manager.SharedMemory((3*64))
                self._data['display_state'] = np.ndarray(
                    (3,),
                    dtype='uint64',
                    buffer=self._data['display_state_mem'].buf
                )
                self._data['display_state'][0] = True
                self._data['display_state'][1:] = 0

            self._data['maze'][:] = self._maze.tiles[:]
            self._data["path"].fill(0)

            self._config = {
                "progress_style": kwargs.get("progress_style", 'bar'),
                "interval": kwargs.get("interval", 1),
                "interval_type": kwargs.get("interval_type", "time"),
                "patience": kwargs.get("patience", 0)
            }

        def _validate_maze(self) -> bool:
            if not isinstance(self._maze, Maze):
                print(f"[error] {self._maze} is not a {Maze}")
                return False

        def _check_unwalked(x, y, path_number) -> bool:
            if (x, y) in self._data["path"][path_number]: return False
            else: return True

        def _update_progress(self):
            
            def _show_progress(self):

                def _node_count():
                    sys.stdout.write(f'\r{self._data["node_depth"]} of {self._data["total_nodes"]} nodes')
                    sys.stdout.flush()

                def _bar():
                    printProgressBar(
                        len(path),
                        self._data["total_nodes"],
                        prefix='Finding path...',
                        suffix='node depth'
                    )

                def _path():
                    current_time = time.time()
                    self.show_path(path)
                    print(
                        f"Steps: {self._data['step_count']:,} ",
                        f"Time: {(time.time()-start_time):.0f} sec",
                        f"\nSteps/second: {int(self._data['step_count']/(current_time-start_time)):,}"
                    )
                
                def _pygame():
                    if not self.display:
                        self.display = Process(
                            target=pygame_display,
                            args=[
                                self._data['maze'].dtype,
                                self._data['maze_mem'],
                                self._data['maze'].shape,
                                self._data['path'].dtype,
                                self._data['path_mem'],
                                self._data['path'].shape,
                                self._data['display_state_mem']
                            ],
                            daemon=True
                        )
                        self.display.start()

                    else:
                        self._data['display_state'][1] = self._data['step_count']
                        self._data['display_state'][2] = (int(
                            self._data['step_count']/(time.time()-start_time)
                        ))
                        formatted_time = time.strftime('%H:%M:%S', time.gmtime())
                        print(
                            f'solving... {formatted_time}',
                            end='\r'
                        )
                
                # call function based on progress style
                {
                    'node_count': _node_count,
                    'bar': _bar,
                    'path': _path,
                    'pygame': _pygame
                }.get(self._config['progress_style'])()

            if self._config["interval_type"] == 'time':
                if time.time() - self._data.get("progress_timer", start_time) \
                    > self._config["interval"]:
                    _show_progress(self)
                    self._data["progress_timer"] = time.time()
            
            elif self._config["interval_type"] == 'step_count':
                if self._data["step_count"] % self._config["interval"] == 0:
                    _show_progress(self)
                

        def _traverse(self,
        coords: tuple,
        path: set
        ) -> bool:
            """Recursively explores the maze, returns True if the end is found,
            returns False when the end cannot be found"""
            # self._data["node_depth"] += 1
            self._data["step_count"] += 1
            path.add(coords)
            
            _update_progress(self)
            if self._config['patience'] and time.time() - start_time > self._config['patience']:
                raise TimeoutError

            if coords == self._maze.end:
                return True

            connections = [c for c in self._data["nodes"][coords] if not c in path]

            # recursively traverse each connection
            for new_coords in connections:
                # set next point in path
                self._data['node_depth'] += 1
                self._data['path'][self._data['node_depth']] = new_coords

                if _traverse(self, new_coords, path):
                    return True

                else:
                    # remove point from path
                    self._data['path'][self._data['node_depth']] = (0, 0)
                    self._data["node_depth"] -= 1
                    path.remove(new_coords)

            # Failed to find path at this point
            return False

        # start pathing the maze
        _setup_solve_data(**kwargs)
        path = set()

        self._data['path'][0] = self._maze.start # first point in path
        try:
            if _traverse(self, self._maze.start, path):
                print("[success] Path found!")

            else:
                print("[fail] No valid path could be found.")
        
        except TimeoutError:
            print('[timeout] took too long to solve')
        
        end_time = time.time()
        self._data["solve_time"] = end_time - start_time


    def create_nodes(self):
        """ Map the maze into a list of nodes"""

        self._data["nodes"] = {}
        i = 0
        map_size = self._maze.x * self._maze.y
        for x in range(self._maze.x):
            for y in range(self._maze.y):
                i += 1
                if self._is_node((x, y)):
                    self._data["nodes"][(x, y)] = []
                printProgressBar(
                    i, map_size, 'Creating nodes...  ',
                    f'{i}/{map_size}', length=50)

        numer_of_nodes = len(self._data["nodes"])
        for i, node in enumerate(self._data["nodes"].keys()):
            self._connect_nodes(node)
            printProgressBar(
                i, numer_of_nodes-1, 'Connecting nodes...',
                f'{i+1}/{numer_of_nodes}', length=50
            )


    def _connect_nodes(self, node: tuple):
        """connect nodes together"""
        x, y = node

        def _down():
            dy = y
            while dy > 0 and self._maze.tiles[x, dy] == 0:
                dy -= 1
                # if self._is_node((x, dy)):
                if (x, dy) in self._data["nodes"]:
                    self._data["nodes"][(x, y)].append((x, dy))
                    break

        def _left():
            dx = x
            while dx > 0 and self._maze.tiles[dx, y] == 0:
                dx -= 1
                # if self._is_node((dx, y)):
                if (dx, y) in self._data["nodes"]:
                    self._data["nodes"][(x, y)].append((dx, y))
                    break

        def _right():
            dx = x
            while dx < self._maze.x-1 and self._maze.tiles[dx, y] == 0:
                dx += 1
                # if self._is_node((dx, y)):
                if (dx, y) in self._data["nodes"]:
                    self._data["nodes"][(x, y)].append((dx, y))
                    break

        def _up():
            dy = y
            while dy < self._maze.y-1 and self._maze.tiles[x, dy] == 0:
                dy += 1
                # if self._is_node((x, dy)):
                if (x, dy) in self._data["nodes"]:
                    self._data["nodes"][(x, y)].append((x, dy))
                    break

        # order determines order of pathing
        _down()
        _left()
        _right()
        _up()

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
                elif (x, y) in self._data['nodes'].keys():
                    line.append('+')
                elif self._maze.tiles[x, y] == 0:
                    line.append(' ')
                else:
                    line.append('e')
            print(' '.join(line))

    def show_path(self, path: tuple = None):
        """Prints to terminal the map with the path :n: drawn on it"""
        if not path is list:
            path = [tuple(p) for p in self._data['path'][:] if tuple(p) != (0, 0)]            

        # tile type to character lookup dict
        characters = {
                0: ' ',
                1: '#',
                2: '2',
                3: '.'
            }

        output = []

        def _draw_path_between_points(a, b, maze):
            # if a is None then b should be the start
            if a is None:
                maze[b] = 3

            else:
                dx = 0
                if b[0] > a[0]: dx = 1
                elif b[0] < a[0]: dx = -1

                dy = 0
                if b[1] > a[1]: dy = 1
                elif b[1] < a[1]: dy = -1

                # paint path onto maze
                x, y = a
                while (x, y) != b:
                    x += dx
                    y += dy
                    maze[x, y] = 3

        if path:
            maze = copy.deepcopy(self._maze.tiles)
            point_a = None
            for point_b in path:
                _draw_path_between_points(point_a, point_b, maze)
                point_a = point_b

            for x in range(maze.shape[0]):
                line = [' ']
                for y in range(maze.shape[1]):
                    if (x, y) == path[-1]: line.append('X')
                    else: line.append(characters[maze[x, y]])
                output.append(' '.join(line))

        if output:
            print('\n'.join(output))

        else:
            print('[error] No path to display')


def pygame_display(
    maze_dtype: np.dtype,
    maze_mem: SharedMemoryManager.SharedMemory,
    maze_shape: tuple,
    path_dtype: np.dtype,
    path_mem: SharedMemoryManager.SharedMemory,
    path_shape: tuple,
    state_mem: SharedMemoryManager.SharedMemory,
    fps: int=244,
    window_size: tuple=(500, 500)):

    import pygame
    from pygame.locals import SRCALPHA
    from pygame.locals import BLEND_ADD, BLEND_SUB, BLEND_MULT, BLEND_MIN, BLEND_MAX

    maze_raw = np.ndarray(maze_shape, dtype=maze_dtype, buffer=maze_mem.buf)
    path_raw = np.ndarray(path_shape, dtype=path_dtype, buffer=path_mem.buf)
    
    state = np.ndarray((3,), dtype='uint64', buffer=state_mem.buf)
    maze = np.array(maze_raw)
    path = np.array(path_raw)

    color = {
        0: (75, 75, 75), # floor
        1: (255, 150, 100), # wall
        2: (255, 255, 0), # error?
        3: (150, 250, 150) # path?
    }

    status_bar_size = (window_size[0], 20)
    maze_surface_size = (window_size[0], window_size[1]-20)

    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(window_size)
    info_surface = pygame.Surface(status_bar_size)
    maze_surface = pygame.Surface(maze_surface_size)
    path_surface = pygame.Surface(maze_surface_size, flags=SRCALPHA)
    tile_size = int(min(maze_surface_size) // max(maze.shape))

    def get_state() -> tuple:
        return tuple(state)

    def _update_maze() -> bool:
        if np.equal(maze, maze_raw).all():
            return False
        else:
            maze[:] = maze_raw[:]
            return True

    def _update_path():
        path[:] = path_raw[:]

    def _draw_tiles():
        for x in range(maze.shape[0]):
            for y in range(maze.shape[1]):
                pygame.draw.rect(
                    maze_surface,
                    color[maze[x, y]],
                    [x*tile_size, y*tile_size, tile_size, tile_size]
                )
    
    def _draw_path():
        pad = int(tile_size // 2)
        points = tuple((x*tile_size+pad, y*tile_size+pad) for x, y, in iter(lambda p=iter(path): tuple(next(p)), (0, 0)))
        path_surface.fill((0, 0, 0, 0))
        if len(points) < 2:
            return
        pygame.draw.aalines(path_surface, color[3], False, points)

    def _draw_info():
        info = get_state()
        current_fps = clock.get_fps()
        fps_font = pygame.font.SysFont('times new roman', 16)
        f_surf = fps_font.render(
            f'FPS: {current_fps:.>6.1f} steps/sec: {info[2]:.>9,d} steps: {info[1]:,}',
            True,
            (200, 200, 200),
            (0, 0, 0)
        )
        info_surface.fill((0, 0, 0))
        info_surface.blit(f_surf, (0, 0))
        

    def _handle_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state[0] = False

    pygame.init()
    screen.fill((0, 0, 0))

    _draw_tiles()

    while state[0] == True:
        _handle_events()
        if _update_maze():
            _draw_tiles()

        _update_path()
        _draw_path()
        _draw_info()

        screen.blit(maze_surface, (0, 0))
        screen.blit(path_surface, (0, 0))
        screen.blit(info_surface, (0, window_size[1]-status_bar_size[1]))

        clock.tick(fps)
        pygame.display.flip()

    pygame.quit()            


if __name__ == "__main__":
    print("maze.py is now running, this is a WIP\n")
    
    while True:
        print('\n')

        test_maze = Maze(10, 10)
        test_maze.display()

        time.sleep(2)

        guy = PathFinder(test_maze)
        print('\n\ncreating nodes...')
        guy.create_nodes()
        guy.display_nodes()
        
        time.sleep(2)

        print('\n\nFinding path...')
        guy.solve(progress_style='path', interval_type='time', interval=1)
        guy.show_path()
        
        print(f'[results] Steps: {guy._data["step_count"]}',
            f'Elapse Time: {guy._data["solve_time"]}'
        )

        time.sleep(5)
