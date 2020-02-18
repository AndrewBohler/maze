from maze import *

import time

# sizes larger than about 15 will take explonentially longer
X_SIZE = 30
Y_SIZE = 30

print("maze.py is now running, this is a WIP\n")
    
while True:
    # Maze class will generate a maze during init
    test_maze = Maze(X_SIZE, Y_SIZE)
    
    # Print the maze to terminal
    test_maze.display()

    time.sleep(2)

    # pass the maze to a pathfinder object
    guy = PathFinder(test_maze)
    print('\n\ncreating nodes...')
    
    # Create a node map to path through
    guy.create_nodes()

    # Print the node map to terminal
    guy.display_nodes()
    print('\n\n')
    
    time.sleep(2)

    print('Finding path...')

    # Attempt to solve the maze (takes a while depending on maze complexity)
    guy.solve(progress_style='path', interval_type='time', interval=1)
    
    # Print the path (if one is found) to terminal
    guy.show_path()
    
    # Print some additonal stats
    print(f'[results] Steps: {guy._data["step_count"]}',
        f'Elapse Time: {guy._data["solve_time"]}'
    )

    time.sleep(5)
