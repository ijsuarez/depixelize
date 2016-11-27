import sys, png
import networkx as nx
import matplotlib.pyplot as plt

'''
Container class to hold pixel data.
'''
class Pixel:
  def __init__(self, y, u, v, coord):
    self.y = y
    self.u = u
    self.v = v
    self.coord = coord

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
    self.reshapeGraph(graph)
    
    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, node_size=5)
    plt.savefig('debug.png')
    
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
            if (i != x or j != y) and (i >= 0 and i < self.width) and (j >= 0 and j < self.height):
              graph.add_edge((x,y),(i,j))

    return graph
    
  '''
  Implements the "Reshaping the Pixel Cells" portion of the algorithm.
  '''
  def reshapeGraph(self, graph):
    self.removeDissimilarEdges(graph)
    self.resolveDiagonalEdges(graph)
  
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
Main method.
'''
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Missing argument: \'python depixelize.py <target_image>\''
    exit()
  depixelizer = Depixelizer()
  depixelizer.depixelize(sys.argv[1])