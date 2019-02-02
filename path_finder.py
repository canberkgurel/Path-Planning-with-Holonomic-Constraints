from waypoint import Waypoint
import numpy as np
from heapq import *


class PathFinder(object):

    def heuristic(self, a, b, hi=0, hj=0):
        heurTemp = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        if (hi != 0) & (hj != 0): heurTemp += 1.4   # some moves have additional cost
        return heurTemp

    def north(self, arr, c, ni=0, nj=0):
        if abs(ni + nj) == 1:
            n_t = 0
        elif abs(ni + nj) == 0:
            n_t = 1
        elif abs(ni + nj) == 2:
            n_t = 3
        n_n = [(-1, 0), (1, 0), (1, -1), (-1, 1), (-1, -1), (1, 1)]  # if the current orientation is 0 then available neighbors are these
        if c != None:
            nr = c[0]  # north _ current row
            nc = c[1]  # north _ current column
            [row, _] = arr.shape
            if (nr != 0):
                if (arr[nr - 1][nc] == 1):
                    n_n = [(-1, 0), (1, 0), (1, -1), (1, 1)]
            if (nr != row - 1):
                if (arr[nr + 1][nc] == 1):
                    n_n = [(-1, 0), (1, 0), (-1, 1), (-1, -1)]
            if ((nr != 0) & (nr != row - 1)):
                if ((arr[nr - 1][nc] == 1) & (arr[nr + 1][nc] == 1)):
                    n_n = [(-1, 0), (1, 0)]
        return [n_n, n_t]

    def east(self, arr, c, ei=0, ej=0):
        e_t = abs(ei + ej)
        e_n = [(0, 1), (0, -1), (1, -1), (-1, 1), (-1, -1), (1, 1)]  # if the current orientation is 1 then available neighbors are these
        if c != None:
            er = c[0]   # east _ current row
            ec = c[1]   # east _ current column
            [_, col] = arr.shape
            if (ec != 0):
                if (arr[er][ec - 1] == 1):
                    e_n = [(0, 1), (0, -1), (-1, 1), (1, 1)]
            if (ec != col - 1):
                if (arr[er][ec + 1] == 1):
                    e_n = [(0, 1), (0, -1), (1, -1), (-1, -1)]
            if ((ec != 0) & (ec != col - 1)):
                if ((arr[er][ec - 1] == 1) & (arr[er][ec + 1] == 1)):
                    e_n = [(0, 1), (0, -1)]
        return [e_n, e_t]

    def south(self, arr, c, si=0, sj=0):
        if abs(si + sj) == 0:
            s_t = 3
        elif abs(si + sj) == 1:
            s_t = 2
        elif abs(si + sj) == 2:
            s_t = 1
        s_n = [(-1, 0), (1, 0), (1, -1), (-1, 1), (-1, -1), (1, 1)]  # if the current orientation is 2 then available neighbors are these
        if c != None:
            sr = c[0]   # south _ current row
            sc = c[1]   # south _ current column
            [row, _] = arr.shape
            if (sr != 0):
                if (arr[sr - 1][sc] == 1):
                    s_n = [(-1, 0), (1, 0), (1, -1), (1, 1)]
            if (sr != row - 1):
                if (arr[sr + 1][sc] == 1):
                    s_n = [(-1, 0), (1, 0), (-1, 1), (-1, -1)]
            if ((sr != 0) & (sr != row - 1)):
                if ((arr[sr - 1][sc] == 1) & (arr[sr + 1][sc] == 1)):
                    s_n = [(-1, 0), (1, 0)]
        return [s_n, s_t]

    def west(self, arr, c, wi=0, wj=0):
        if abs(wi + wj) == 0:
            w_t = 2
        elif abs(wi + wj) == 1:
            w_t = 3
        elif abs(wi + wj) == 2:
            w_t = 0
        w_n = [(0, 1), (0, -1), (1, -1), (-1, 1), (-1, -1), (1, 1)]  # if the current orientation is 3 then available neighbors are these
        if c != None:
            wr = c[0]   # west _ current row
            wc = c[1]   # west _ current column
            [_, col] = arr.shape
            if (wc != 0):
                if (arr[wr][wc - 1] == 1):
                    w_n = [(0, 1), (0, -1), (-1, 1), (1, 1)]
            if (wc != col - 1):
                if (arr[wr][wc + 1] == 1):
                    w_n = [(0, 1), (0, -1), (1, -1), (-1, -1)]
            if ((wc != 0) & (wc != col - 1)):
                if ((arr[wr][wc - 1] == 1) & (arr[wr][wc + 1] == 1)):
                    w_n = [(0, 1), (0, -1)]
        return [w_n, w_t]

    def get_path(self, grid, start_wp, end_wp):
        array = grid.astype(int)    # convert from Cartesian Coordinates to matrix representation
        array = np.flipud(array)    # flip the array up side down
        [numrow, _] = array.shape
        numrow -= 1
        start = tuple([numrow - start_wp._y, start_wp._x, start_wp._orientation])   # convert the waypoints to tuple
        goal = tuple([numrow - end_wp._y, end_wp._x, end_wp._orientation])


        """Returns a list of Waypoints from the start Waypoint to the end Waypoint.
:
        :param grid: Grid is a 2D numpy ndarray of boolean values. grid[x, y] == True if the cell contains an obstacle.
        The grid dimensions are exposed via grid.shape
        :param start_wp: The Waypoint that the path should start from.
        :param end_wp: The Waypoint that the path should end on.
        :return: The path from the start waypoint to the end waypoint that follows the movement model without going
        off the grid or intersecting an obstacle.
        :rtype: A list of Waypoints.

        More documentation at
        https://docs.google.com/document/d/1b30L2LeKyMjO5rBeCui38j_HSUYgEGWXrwSRjB7AnYs/edit?usp=sharing
        """

        try:
            if array[start[0], start[1]]:
                raise NameError('Start destination is inside an obstacle.')
            elif array[goal[0], goal[1]]:
                raise NameError('Goal destination is inside an obstacle.')
        except NameError:
            print('Error Occurred! :')
            raise

        direcDict = {0: self.north, 1: self.east, 2: self.south, 3: self.west}
        goalDirection = goal[2]
        close_set = set()
        came_from = {}
        gscore = {start: 0}     # A star G score
        fscore = {start: self.heuristic(start, goal)}  # A star F score
        oheap = []

        heappush(oheap, (fscore[start], start))

        while oheap:

            current = heappop(oheap)[1]

            if (current[0] == goal[0]) & (current[1] == goal[1]) & (current[2] == goalDirection):
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                data.reverse()
                list = []
                for i in range(len(data)):      # convert back to the Cartesian Coordinates representation
                    temp_zero = data[i][0]
                    temp_one = data[i][1]
                    temp_two = data[i][2]
                    list.append(Waypoint(temp_one, numrow - temp_zero, temp_two))
                print(list)

                return list  # [start_wp, Waypoint(5, 6, 0), Waypoint(5, 7, 0), end_wp]

            close_set.add(current)
            [neighbors, _] = direcDict[current[2]](array, current)

            for i, j in neighbors:  # explore the map for the available set of neighbors
                [_, neigh_orien] = direcDict[current[2]]([], None, i, j)
                neighbor = current[0] + i, current[1] + j, neigh_orien
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor, i, j)
                if 0 <= neighbor[0] < array.shape[0]:
                    if 0 <= neighbor[1] < array.shape[1]:
                        if array[neighbor[0]][neighbor[1]] == 1:
                            continue
                    else:
                        # array bound y walls
                        continue
                else:
                    # array bound x walls
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue

                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heappush(oheap, (fscore[neighbor], neighbor))

        return False

        #print "EDIT HERE"

