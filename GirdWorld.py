import numpy as np 
import random
import pickle
import os 

class GridWorld: 
    """
    Creates a grid that is a 2D array
    0 = unblocked cells 
    1 = blocked cells 
    """
    def __init__(self, size=101):
        self.grid = np.zeros((size,size), dtype=np.int8)
        self.size = size
        
    def generate(self,seed=None):
        if seed is not None: 
            random.seed(seed)
            np.random.seed(seed)
        
        empty = self.grid.copy()
        visited = np.zeros((self.size,self.size), dtype=bool)
        stack = []
    
        current_row = np.random.randint(self.size)
        current_col = np.random.randint(self.size)
    
        visited[current_row,current_col] = True 
        empty[current_row,current_col] = 0 
        stack.append((current_row,current_col))
    
        unblocked_blocked = [0,1]
        weights = [0.7,0.3]
    
            # Add iteration counter and limit
        iteration = 0
        max_iterations = self.size * self.size * 100  # Safety limit
    
        while not visited.all() and iteration < max_iterations:
            iteration += 1
        
        # Print progress every 1000 iterations
            neighbors = self.get_neighbors(current_row, current_col)
            unvisited = [n for n in neighbors if not visited[n[0], n[1]]]
        
            if len(unvisited) > 0:
                current_row, current_col = self.next_neighbor(unvisited)
                visited[current_row,current_col] = True 
                empty[current_row, current_col] = random.choices(unblocked_blocked,weights, k=1)[0]  
                if empty[current_row,current_col] == 0:
                    stack.append((current_row,current_col))
            else: 
                #handle dead end back track, avoids infinite loop 
                found_valid_cell = False
                while len(stack) > 0: 
                    current_row, current_col = stack.pop()
                    neighbors = self.get_neighbors(current_row,current_col)
                    unvisited = [n for n in neighbors if not visited[n[0], n[1]]]

                    if len(unvisited) > 0: 
                        found_valid_cell = True 
                        break
                   
                if not found_valid_cell: 
                    unvisited_position = np.argwhere(visited==False)
                    if len(unvisited_position) > 0:
                        current_row, current_col = tuple(unvisited_position[0])
                        visited[current_row, current_col] = True
                        empty[current_row,current_col] = 0
                        stack.append((current_row,current_col))
        self.grid = empty            
        return self.grid 
    
    def next_neighbor(self,unvisited):
        return random.choice(unvisited)


    #Checks if the neighboring point is on the grid 
    def is_on_grid(self,row, col):
        return 0 <= row < self.size and 0 <= col < self.size 
    
    #get the neighbors of our current position on the grid 
    def get_neighbors(self, row,col):
       neighbors = []

       directions = [
          (-1,0), #North
          (1,0), #South
          (0,-1), #West
          (0,1) #East
       ]

       for dr, dc in directions:
           new_row = dr + row
           new_col = dc + col

           if self.is_on_grid(new_row,new_col):
               neighbors.append((new_row, new_col))

       return neighbors   

    def save(self,filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f: 
            pickle.dump(self.grid, f)
    def load(self,filename):
        with open(filename, 'rb') as f:
            self.grid = pickle.load(f)
        self.size = self.grid.shape[0]

def generate_all_grids():
    #generate and save 50 grid worlds 
    os.makedirs('grids', exist_ok=True)
    
    for i in range(50):
        gw = GridWorld(size=101)
        gw.generate(seed=i)
        gw.save(f'grids/grid_{i:02d}.pkl')

        blocked = np.sum(gw.grid)
        total = gw.size * gw.size

if __name__ == "__main__":
    generate_all_grids()

