from maze import *

import time

X_SIZE = 30
Y_SIZE = 30

print("maze.py is now running, this is a WIP\n")
    
while True:
    test_maze = Maze(X_SIZE, Y_SIZE)
    test_maze.display()

    time.sleep(2)

    guy = PathFinder(test_maze)
    print('\n\ncreating nodes...')
    guy.create_nodes()
    guy.display_nodes()
    print('\n\n')
    
    time.sleep(2)

    print('Finding path...')
    guy.solve(progress_style='path', interval_type='time', interval=1)
    guy.show_path()
    
    print(f'[results] Steps: {guy._data["step_count"]}',
        f'Elapse Time: {guy._data["solve_time"]}'
    )

    time.sleep(5)