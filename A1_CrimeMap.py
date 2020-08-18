# -------------------------------------------------------
# Assignment 1
# Written by Jordan Hum - 40095876
# For COMP 472 Section IX – Summer 2020
# --------------------------------------------------------

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import heapq as hq
import time

def distance(a, b):
    # Find the distance between two points
    #   Also used as the heuristic function
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def validate(i, j, current):
    # Move right, below and above bin cannot be blocked
    if i == 1 and j == 0:
        if totalCrimeList[current] == 1 and totalCrimeList[current[0]][current[1]-1] == 1:
            return True

    # Move bottom right, diagonal bin cannot be blocked
    if i == 1 and j == -1:
        if totalCrimeList[current[0]][current[1]-1] == 1:
            return True

    # Move down, left and right bin cannot be blocked
    if i == 0 and j == -1:
        if totalCrimeList[current[0]][current[1]-1] == 1 and totalCrimeList[current[0]-1][current[1]-1] == 1:
            return True

    # Move bottom left, diagonal bin cannot be blocked
    if i == -1 and j == -1:
        if totalCrimeList[current[0]-1][current[1]-1] == 1:
            return True

    # Move left, below and above bin cannot be blocked
    if i == -1 and j == 0:
        if totalCrimeList[current[0]-1][current[1]] == 1 and totalCrimeList[current[0]-1][current[1]-1] == 1:
            return True

    # Move top left, diagonal bin cannot be blocked
    if i == -1 and j == 1:
        if totalCrimeList[current[0]-1][current[1]] == 1:
            return True

    # Move up, left and right bin cannot be blocked
    if i == 0 and j == 1:
        if totalCrimeList[current[0]-1][current[1]] == 1 and totalCrimeList[current] == 1:
            return True

    # Move top right, diagonal bin cannot be blocked
    if i == 1 and j == 1:
        if totalCrimeList[current] == 1:
            return True

def findPath(grid, start, end):
    moves = [(1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1)]

    startPoint = grid[(start[0], start[1])]
    endPoint = grid[(end[0], end[1])]

    openList = []
    closedList = []
    parentNodes = {}
    gScores = {}
    fScores = {}

    # Initialize start score for f(n) and g(n), add start to priority queue
    gScores[start] = 0
    hScore = distance(start, end)
    fScores[start] = hScore
    hq.heappush(openList, (fScores[start], start))

    # Check each point in the openList
    while openList:
        # Retrieve point to check all next possible moves and add to closedList
        current = hq.heappop(openList)[1]
        closedList.append(current)

        # Check if algorithm takes longer than 10 seconds
        if time.time() - startTime >= 10:
            print("Time is up. The optimal path is not found.")
            raise SystemExit(0)

        # Check if the end point has been found
        if current == end:
            # Add end point to the solution path
            path.append(endPoint)
            point = parentNodes[end]

            # Trace back each parent node from the end to the beginning
            while point != start:
                path.append(grid[(point[0], point[1])])
                point = parentNodes[point]

            # Add start point to the solution path
            path.append(startPoint)

            # Reverse the solution path since it's traced it backwards
            return path.reverse()

        # Check all next possible moves from the current point
        for i, j in moves:
            nextMove = (current[0] + i, current[1] + j)

            # Check if a next move is valid
            if 0 >= nextMove[0] or nextMove[0] > numBin-1 or 0 >= nextMove[1] or nextMove[1] > numBin-1 or nextMove in closedList:
                # Invalid move or already checked, skip
                continue

            # Check for blocked spaces when traversing along edges
            if validate(i, j, current):
                continue

            # Calculate g(n) for next move
            gScore = gScores[current] + distance(current, nextMove)

            # Check if a g(n) is declared for next move
            #   or if for the same point g(n) is better (a better path is found)
            if nextMove not in gScores or gScore < gScores[nextMove]:
                # Update parent of the point
                parentNodes[nextMove] = current

                # Update f(n) and g(n)
                gScores[nextMove] = gScore
                hScore = distance(nextMove, end)
                fScores[nextMove] = gScore + hScore

                # Add point to the priority queue
                hq.heappush(openList, (fScores[nextMove], nextMove))

def plotSolution():
    if path:
        # Plot solution grid
        plt.figure(2)
        plt.axes().set_aspect("equal")
        plt.hist2d(longitude, latitude, bins=numBin, vmin=percentile, vmax=(percentile + 0.1))
        plt.xticks(np.arange(xMin, xMax, gridSize), rotation=90)
        plt.yticks(np.arange(yMin, yMax, gridSize), rotation=0)

        # Plot start / end points
        plt.plot(grid[startBin][0], grid[startBin][1], 'o', markersize=10, color='g')
        plt.plot(grid[endBin][0], grid[endBin][1], 'o', markersize=10, color='r')

        xCoordinates = []
        yCoordinates = []

        # Draw solution path
        for i in range(len(path)):
            xCoordinates.append(path[i][0])
            yCoordinates.append(path[i][1])
            plt.plot(xCoordinates, yCoordinates, linewidth=2, color='w')
    else:
        print("Due to blocks, no path is found. Please change the map and try again.")

#######################################################################################################################
# Boundaries of the grid
xMin = -73.590
xMax = -73.550
yMin = 45.490
yMax = 45.530

# Read shape file
gdf = gpd.read_file('Shape/crime_dt.shp')
points = gdf['geometry']
longitude = np.array(points.x)
latitude = np.array(points.y)

# User input for grid size and threshold
gridSize = float(input("Enter grid size: "))
threshold = int(input("Enter threshold: "))

# Calculate grid size
numBin = int(np.abs(np.ceil((xMin - xMax) / gridSize)))
print("Map size of ", str(numBin), "x", str(numBin))

# Plot crime rate grid
plt.figure(1)
plt.axes().set_aspect("equal")
totalCrimeList, xEdges, yEdges, img = plt.hist2d(longitude, latitude, bins=numBin)
plt.xticks(np.arange(xMin, xMax, gridSize), rotation=90)
plt.yticks(np.arange(yMin, yMax, gridSize), rotation=0)

# Display total crime per grid
for i in range(len(totalCrimeList)):
    for j in range(len(totalCrimeList[i])):
        plt.text((xEdges[i]+(gridSize/2)), (yEdges[j]+(gridSize/2)), int(totalCrimeList[i][j]), color="w", ha="center", va="center")

# Calculate percentile for the desired threshold input
#   Gives the smallest value of the threshold
percentile = np.percentile(totalCrimeList, threshold)
print("Percentile: ", str(percentile))

# Create 2D histogram of crime rate per bin
crimePerBin = np.histogram2d(longitude, latitude, bins=numBin)

# Calculate mean
mean = np.mean(totalCrimeList)
print("Mean: ", str(mean))

# Calculate standard deviation
std = np.std(totalCrimeList)
print("Standard deviation: ", str(std))

# Generate dictionary of bin indexes to coordinates
#   Converts threshold grid to 1s and 0s for simplicity
grid = {}
for i in range(len(totalCrimeList)):
    for j in range(len(totalCrimeList[i])):
        grid[(i, j)] = (xEdges[i], yEdges[j])
        if crimePerBin[0][i][j] >= percentile:
            totalCrimeList[i][j] = 1
        else:
            totalCrimeList[i][j] = 0

# Start / End points
print("Enter a grid start location from ( 0 , 0 ) to (", (numBin-1), ", ", (numBin-1), "): ")
x, y = input().split()
startBin = (int(x), int(y))
print("Enter a grid end location from ( 1 , 1 ) to (", (numBin-1), ", ", (numBin-1), "): ")
x, y = input().split()
endBin = (int(x), int(y))

# Find and plot shortest path
startTime = time.time()
path = []
findPath(grid, startBin, endBin)
plotSolution()
endTime = time.time()

# Display execution time
print("Execution time: ", str(endTime - startTime), "seconds")

# Terminate program
print("The program has terminated.")
plt.show()
