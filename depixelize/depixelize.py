import sys, math, png
import networkx as nx
import matplotlib.pyplot as plt

from pyhull import qconvex

'''
Container class to hold pixel data.
'''
class Pixel:
  def __init__(self, y, u, v, coord):
    self.y = y
    self.u = u
    self.v = v
    self.coord = coord
    self.voronoiPts = []

'''
Main class for depixelizer functions.
'''
class Depixelizer:

  '''
  Constructor for Depixelizer object.
  '''
  def __init__(self):
    self.pixelData = {}
    self.width = 0
    self.height = 0
    
  '''
  Main function to be called.
  '''
  def depixelize(self, target):
    graph = self.initializeGraph(target)
    voronoi = self.reshapeGraph(graph)

    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, node_size=5)
    plt.savefig('graph.png')

    plt.clf()
    
    pos = nx.get_node_attributes(voronoi, 'pos')
    nx.draw(voronoi, pos, node_size=5)
    plt.savefig('voronoi.png')
    
  '''
  Takes a PNG image and turns it into a graph with YUV data.
  '''
  def initializeGraph(self, target):
    rgbData = self.readPNG(target)
    yuvData = self.convertRGBtoYUV(rgbData)
    graphData = self.convertYUVtoGraph(yuvData)
    
    return graphData
    
  '''
  Reads in the PNG file and saves the width and height.
  '''
  def readPNG(self, image):
    r = png.Reader(image)
    rgbData = r.asRGBA8()
    
    self.width = rgbData[0]
    self.height = rgbData[1]
    
    return rgbData
    
  '''
  Converts the RGB data to YUV data.
  '''
  def convertRGBtoYUV(self, rgbData):
    # assuming 4 bands of data, RGBA, ignore alpha channel
    yuvData = []
    for row in rgbData[2]:
      yuvRow = []
      for i in range(0, len(row), 4):
        yuvRow.append((0.257 * row[i]) + (0.504 * row[i+1]) + (0.098 * row[i+2]) + 16)   # y-value
        yuvRow.append(-(0.148 * row[i]) - (0.291 * row[i+1]) + (0.439 * row[i+2]) + 128) # u-value
        yuvRow.append((0.439 * row[i]) - (0.368 * row[i+1]) - (0.071 * row[i+2]) + 128)  # v-value
      yuvData.append(yuvRow)
    return yuvData
    
  '''
  Creates a dictionary of pixel coordinate to YUV data and uses the coordinates as keys for a networkx graph.
  Creates edges for all adjacent nodes.
  '''
  def convertYUVtoGraph(self, yuvData):
    graph = nx.Graph()
    
    for row in range(len(yuvData)):
      for col in range(0, len(yuvData[row]), 3):
        # make coordinates follow (x, y) format
        pixel = Pixel(yuvData[row][col], yuvData[row][col+1], yuvData[row][col+2], (col/3, row))
        self.pixelData[(col/3, row)] = pixel
        graph.add_node((col/3, row), pos=(col/3, self.height-row))
    
    for y in range(self.height):
      for x in range(self.width):
        for j in range(y-1,y+2):
          for i in range(x-1,x+2):
            if (i != x or j != y) and self.inRange((i, j)):
              graph.add_edge((x,y),(i,j))

    return graph
    
  def inRange(self, coord):
    return (coord[0] >= 0 and coord[0] < self.width) and (coord[1] >= 0 and coord[1] < self.height)
    
  '''
  Implements the "Reshaping the Pixel Cells" portion of the algorithm.
  '''
  def reshapeGraph(self, graph):
    self.removeDissimilarEdges(graph)
    self.resolveDiagonalEdges(graph)
    return self.computeVoronoi(graph)
  
  '''
  Removes edges where the difference between pixels is large.
  '''
  def removeDissimilarEdges(self, graph):
    for node in graph.nodes():
      nodeData = self.pixelData[node]
      neighbors = graph.neighbors(node)
      for neighbor in neighbors:
        neighborData = self.pixelData[neighbor]
        if (abs(nodeData.y - neighborData.y) > 48/255 or abs(nodeData.u - neighborData.u) > 7/255 or
            abs(nodeData.v - neighborData.v) > 6/255):
          graph.remove_edge(node, neighbor)
          
  '''
  Uses a sliding 2x2 window to examine pixel blocks and remove the necessary diagonals.
  '''
  def resolveDiagonalEdges(self, graph):
    for y in range(self.height-1):
      for x in range(self.width-1):
        if self.fullyConnected(graph, (x, y)):
          graph.remove_edge((x,y), (x+1,y+1))
          graph.remove_edge((x+1,y), (x,y+1))
        elif self.onlyDiagonalConnections(graph, (x, y)):
          w1, w2 = self.computeWeights(graph, (x, y))
          if w1 < w2:
            graph.remove_edge((x,y), (x+1,y+1))
          elif w2 < w1:
            graph.remove_edge((x+1,y), (x,y+1))
          else:
            graph.remove_edge((x,y), (x+1,y+1))
            graph.remove_edge((x+1,y), (x,y+1))
        
  '''
  Helper function to determine if a 2x2 window of pixels is fully connected.
  '''
  def fullyConnected(self, graph, coord):
    c1 = coord
    c2 = (coord[0]+1, coord[1])
    c3 = (coord[0], coord[1]+1)
    c4 = (coord[0]+1, coord[1]+1)
    
    return (self.nodesConnected(graph, c1, c2) and self.nodesConnected(graph, c1, c3) and
            self.nodesConnected(graph, c1, c4) and self.nodesConnected(graph, c2, c3) and
            self.nodesConnected(graph, c2, c4) and self.nodesConnected(graph, c3, c4))
            
  '''
  Helper function to determine if a 2x2 window of pixels has only diagonal connections.
  '''
  def onlyDiagonalConnections(self, graph, coord):
    c1 = coord
    c2 = (coord[0]+1, coord[1])
    c3 = (coord[0], coord[1]+1)
    c4 = (coord[0]+1, coord[1]+1)
    
    return (self.nodesConnected(graph, c1, c4) and self.nodesConnected(graph, c2, c3) and not
            (self.nodesConnected(graph, c1, c2) or self.nodesConnected(graph, c1, c3) or
            self.nodesConnected(graph, c2, c4) or self.nodesConnected(graph, c3, c4)))
  
  '''
  Helper function to determine if an edge exists between two pixel nodes.
  '''
  def nodesConnected(self, graph, u, v):
    return u in graph.neighbors(v)
    
  '''
  Helper function to return the weights of the diagonal edges in a 2x2 window.
  '''
  def computeWeights(self, graph, coord):
    c1 = coord
    c2 = (coord[0]+1, coord[1])
    c3 = (coord[0], coord[1]+1)
    c4 = (coord[0]+1, coord[1]+1)
    
    w1 = self.curveWeight(graph, c1, c4) + self.sparseWeight(graph, c1, c2) + self.islandWeight(graph, c1, c4)
    w2 = self.curveWeight(graph, c2, c3) + self.sparseWeight(graph, c2, c1) + self.islandWeight(graph, c2, c3)
    
    return w1, w2
    
  '''
  Heuristic to compute the weight based on the length of the curve.
  '''
  def curveWeight(self, graph, c1, c2):
    visited = set([c1, c2])
    stack = [c1, c2]
    
    weight = 1
    while len(stack) != 0:
      node = stack.pop()
      neighbors = graph.neighbors(node)
      
      if len(neighbors) == 2:
        weight += 1
        for neighbor in neighbors:
          if neighbor not in visited:
            visited.add(neighbor)
            stack.append(neighbor)
  
    return weight
    
  '''
  Heuristic to compute the weight based on the difference of component connectivity within an 8x8 window.
  '''
  def sparseWeight(self, graph, c1, c2):
    xMin = min(c1[0], c2[0]) - 3
    xMax = max(c1[0], c2[0]) + 3
    yMin = c1[1] - 3
    yMax = c1[1] + 4
    
    stack1 = [c1]
    stack2 = [c2]
    
    visited1 = set()
    visited2 = set()
    
    weight1 = 0
    while len(stack1) != 0:
      node = stack1.pop()
      weight1 += 1
      
      for neighbor in graph.neighbors(node):
        if neighbor in visited1 or neighbor[0] < xMin or neighbor[0] > xMax or neighbor[1] < yMin or neighbor[1] > yMax:
          continue
        visited1.add(neighbor)
        stack1.append(neighbor)
        
    weight2 = 0
    while len(stack2) != 0:
      node = stack2.pop()
      weight2 += 1
      
      for neighbor in graph.neighbors(node):
        if neighbor in visited2 or neighbor[0] < xMin or neighbor[0] > xMax or neighbor[1] < yMin or neighbor[1] > yMax:
          continue
        visited2.add(neighbor)
        stack2.append(neighbor)
        
    return weight2 - weight1
    
  '''
  Heuristic to compute the weight based on if one of the nodes is on an island.
  '''
  def islandWeight(self, graph, c1, c2):
    if len(graph.neighbors(c1)) == 1 or len(graph.neighbors(c2)) == 1:
      return 5
    return 0
    
  '''
  For each pixel, examines the adjacent and corner nodes (eight total) and computes the Voronoi cell. Northwest is (0,0)
  '''
  def computeVoronoi(self, graph):
    voronoi = nx.Graph()
    
    for y in range(self.height):
      for x in range(self.width):     
        pixel = self.pixelData[(x,y)]
        self.findVoronoiPts(graph, pixel)
        hull = [point.split() for point in qconvex('p', pixel.voronoiPts)[2:]]
        hull = [(float(coord[0]),float(coord[1])) for coord in hull]
        pixel.voronoiPts = sorted(hull, key=lambda p: (self.angle(p, (pixel.coord[0]+0.5, pixel.coord[1]+0.5))))
        self.addVoronoiPts(voronoi, pixel)

    return voronoi

  def angle(self, point, center):
    return math.atan2(point[1]- center[1], point[0] - center[0])

  def addVoronoiPts(self, voronoi, pixel):
    for point in pixel.voronoiPts:
      voronoi.add_node(point, pos=(point[0], self.height-point[1]))
    for i in range(len(pixel.voronoiPts)):
      voronoi.add_edge(pixel.voronoiPts[i], pixel.voronoiPts[(i-1)%len(pixel.voronoiPts)])
      voronoi.add_edge(pixel.voronoiPts[i], pixel.voronoiPts[(i+1)%len(pixel.voronoiPts)])
        
  def findVoronoiPts(self, graph, pixel):
    pixelNeighbors = graph.neighbors(pixel.coord)
    
    x, y = pixel.coord
    center = (x + 0.5, y + 0.5)
    
    north = (x, y-1)
    south = (x, y+1)
    west = (x-1, y)
    east = (x+1, y)
    
    if self.inRange(north):
      if north not in pixelNeighbors:
        pixel.voronoiPts.append((center[0], center[1] - 0.25))
    else:
      pixel.voronoiPts.append((center[0], center[1] - 0.5))
      
    if self.inRange(south):
      if south not in pixelNeighbors:
        pixel.voronoiPts.append((center[0], center[1] + 0.25))
    else:
      pixel.voronoiPts.append((center[0], center[1] + 0.5))
      
    if self.inRange(west):
      if west not in pixelNeighbors:
        pixel.voronoiPts.append((center[0] - 0.25, center[1]))
    else:
      pixel.voronoiPts.append((center[0] - 0.5, center[1]))
      
    if self.inRange(east):
      if east not in pixelNeighbors:
        pixel.voronoiPts.append((center[0] + 0.25, center[1]))
    else:
      pixel.voronoiPts.append((center[0] + 0.5, center[1]))
      
    nw = (x-1, y-1)
    sw = (x-1, y+1)
    ne = (x+1, y-1)
    se = (x+1, y+1)
    
    self.findDiagonalVoronoiPts(graph, pixel, pixelNeighbors, center, nw, north, west)
    self.findDiagonalVoronoiPts(graph, pixel, pixelNeighbors, center, sw, south, west)
    self.findDiagonalVoronoiPts(graph, pixel, pixelNeighbors, center, ne, north, east)
    self.findDiagonalVoronoiPts(graph, pixel, pixelNeighbors, center, se, south, east)
          
  def findDiagonalVoronoiPts(self, graph, pixel, pixelNeighbors, center, diagonal, vert, horiz):
    horizSign = -1 if horiz[0] < center[0] else 1
    vertSign = -1 if vert[1] < center[1] else 1
    
    horizNeighbor = self.inRange(horiz) and horiz in pixelNeighbors
    vertNeighbor = self.inRange(vert) and vert in pixelNeighbors
    
    if self.inRange(diagonal):
      if diagonal in pixelNeighbors:
        pixel.voronoiPts.append((center[0] + (horizSign * 0.75), center[1] + (vertSign * 0.25)))
        pixel.voronoiPts.append((center[0] + (horizSign * 0.25), center[1] + (vertSign * 0.75)))
        if (horizNeighbor and not vertNeighbor) or (vertNeighbor and not horizNeighbor):
          pixel.voronoiPts.append((center[0] + (horizSign * 0.5), center[1] + (vertSign * 0.5)))
      else:
        if horiz in graph.neighbors(vert):
          pixel.voronoiPts.append((center[0] + (horizSign * 0.25), center[1] + (vertSign * 0.25)))
        else:
          pixel.voronoiPts.append((center[0] + (horizSign * 0.5), center[1] + (vertSign * 0.5)))
    else:
      pixel.voronoiPts.append((center[0] + (horizSign * 0.5), center[1] + (vertSign * 0.5)))

'''
Main method.
'''
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Missing argument: \'python depixelize.py <target_image>\''
    exit()
  depixelizer = Depixelizer()
  depixelizer.depixelize(sys.argv[1])
