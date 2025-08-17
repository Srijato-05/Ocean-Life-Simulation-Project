# src/utils/spatial_hash.py

from collections import defaultdict

class SpatialHash:
    """
    A data structure for efficiently querying objects in a 2D or 3D space.
    It works by dividing the space into a grid of cells and storing a list
    of object indices contained within each cell.
    """
    def __init__(self, width, height, depth):
        self.hash = defaultdict(list)
        self.width = width
        self.height = height
        self.depth = depth

    def clear(self):
        """Clears the hash, typically at the start of a new simulation tick."""
        self.hash.clear()

    def insert(self, agent_index, position):
        """
        Inserts an agent's index into the hash based on its position.

        Args:
            agent_index (int): The index of the agent in the main NumPy arrays.
            position (np.array): The [x, y, z] position of the agent.
        """
        key = (int(position[0]), int(position[1]), int(position[2]))
        self.hash[key].append(agent_index)

    def get_neighbors(self, agent_index, position, radius):
        """
        Gets the indices of all agents within a given radius of a central agent.
        """
        neighbor_indices = []
        x, y, z = int(position[0]), int(position[1]), int(position[2])

        x_min = max(0, x - radius)
        x_max = min(self.width - 1, x + radius)
        y_min = max(0, y - radius)
        y_max = min(self.height - 1, y + radius)
        z_min = max(0, z - radius)
        z_max = min(self.depth - 1, z + radius)

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                for k in range(z_min, z_max + 1):
                    key = (i, j, k)
                    if key in self.hash:
                        neighbor_indices.extend(self.hash[key])
        
        if agent_index in neighbor_indices:
            neighbor_indices.remove(agent_index)
            
        return neighbor_indices

    def get_indices_in_cell(self, x, y, z):
        """
        Returns a list of all agent indices in a specific cell.
        """
        key = (int(x), int(y), int(z))
        return self.hash.get(key, [])