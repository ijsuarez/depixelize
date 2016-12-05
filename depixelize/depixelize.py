import sys, math, png
import networkx as nx
import matplotlib.pyplot as plt
import svgwrite
import bspline

from pyhull import qconvex

'''
Container class to hold pixel data.
'''
class Pixel:
  def __init__(self, r, g, b, y, u, v, coord):
    self.r = r
    self.g = g
    self.b = b
    self.y = y
    self.u = u
    self.v = v
    self.coord = coord
    self.voronoiPts = []

  def rgbData(self):
    return (self.r, self.g, self.b)

class VisibleEdge:
  def __init__(self, pts):
    self.pts = pts
    self.bspline = None

  def __getitem__(self, key):
    return self.pts[key]

'''
Main class for depixelizer functions.
'''
class Depixelizer:

  '''
  Constructor for Depixelizer object.
  '''
  def __init__(self):
    self.pixelData = {}
    self.pixelMapping = {}
    self.VEmap = {}
    self.width = 0
    self.height = 0
    
  '''
  Main function to be called.
  '''
  def depixelize(self, target):
    graph = self.initializeGraph(target)
    voronoi = self.reshapeGraph(graph)
    visibleEdges = self.generateVisibleEdges(voronoi)
    self.resolveTJunctions(visibleEdges)
    
    edgeGraph = nx.Graph()
    for edge in visibleEdges:
      for i in range(len(edge.pts)-1):
        edgeGraph.add_node(edge[i], pos=(edge[i][0], self.height-edge[i][1]))
        edgeGraph.add_edge(edge[i], edge[i+1])
      edgeGraph.add_node(edge[-1], pos=(edge[-1][0], self.height-edge[-1][1]))

    #self.computeSplines(visibleEdges)
    self.render(visibleEdges)

    pos = nx.get_node_attributes(graph, 'pos')
    nx.draw(graph, pos, node_size=5)
    plt.savefig('graph.png')

    plt.clf()
    
    pos = nx.get_node_attributes(voronoi, 'pos')
    nx.draw(voronoi, pos, node_size=5)
    plt.savefig('voronoi.png')

    plt.clf()

    pos = nx.get_node_attributes(edgeGraph, 'pos')
    nx.draw(edgeGraph, pos, node_size=5)
    plt.savefig('visible.png')
    
  '''
  Takes a PNG image and turns it into a graph with YUV data.
  '''
  def initializeGraph(self, target):
    rgbData = self.readPNG(target)
    rgbData, yuvData = self.convertRGBtoYUV(rgbData)
    graphData = self.convertToGraph(rgbData, yuvData)
    
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
    rgbDataCopy = []
    yuvData = []
    for row in rgbData[2]:
      rgbRow = []
      yuvRow = []
      for i in range(0, len(row), 4):
        rgbRow.append(row[i])
        rgbRow.append(row[i+1])
        rgbRow.append(row[i+2])
        yuvRow.append((0.257 * row[i]) + (0.504 * row[i+1]) + (0.098 * row[i+2]) + 16)   # y-value
        yuvRow.append(-(0.148 * row[i]) - (0.291 * row[i+1]) + (0.439 * row[i+2]) + 128) # u-value
        yuvRow.append((0.439 * row[i]) - (0.368 * row[i+1]) - (0.071 * row[i+2]) + 128)  # v-value
      rgbDataCopy.append(rgbRow)
      yuvData.append(yuvRow)
    return rgbDataCopy, yuvData
    
  '''
  Creates a dictionary of pixel coordinate to YUV data and uses the coordinates as keys for a networkx graph.
  Creates edges for all adjacent nodes.
  '''
  def convertToGraph(self, rgbData, yuvData):
    graph = nx.Graph()
    
    for row in range(self.height):
      for col in range(self.width):
        # make coordinates follow (x, y) format
        pixel = Pixel(rgbData[row][(col*3)], rgbData[row][(col*3)+1], rgbData[row][(col*3)+2],
                      yuvData[row][(col*3)], yuvData[row][(col*3)+1], yuvData[row][(col*3)+2], (col, row))
        self.pixelData[(col, row)] = pixel
        graph.add_node((col, row), pos=(col, self.height-row))
    
    for y in range(self.height):
      for x in range(self.width):
        for j in range(y-1,y+2):
          for i in range(x-1,x+2):
            if (i != x or j != y) and self.inRange((i, j)):
              graph.add_edge((x,y),(i,j))

    return graph
  
  '''
  Helper function to check if a given coordinate is within the proper width/height.
  '''
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
        if self.pixelIsDissimilar(nodeData, neighborData):
          graph.remove_edge(node, neighbor)

  def pixelIsDissimilar(self, nodeData, neighborData):
    return (abs(nodeData.y - neighborData.y) > 48/255 or abs(nodeData.u - neighborData.u) > 7/255 or
            abs(nodeData.v - neighborData.v) > 6/255)
          
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
  For each pixel, examines the adjacent and corner nodes (eight total) and computes the Voronoi cell. Northwest is (0,0).
  The convex hull is computed using the pyhull library. It conveniently provides a decent estimate of the
  Voronoi cell.
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

  '''
  Helper function to sort the convex hull points to be a circular array.
  '''
  def angle(self, point, center):
    return math.atan2(point[1]- center[1], point[0] - center[0])

  '''
  For a given pixel, adds all the Voronoi points into the Voronoi graph.
  '''
  def addVoronoiPts(self, voronoi, pixel):
    for point in pixel.voronoiPts:
      if point not in self.pixelMapping:
        self.pixelMapping[point] = set()
      self.pixelMapping[point].add(pixel)
      voronoi.add_node(point, pos=(point[0], self.height-point[1]))
    for i in range(len(pixel.voronoiPts)):
      voronoi.add_edge(pixel.voronoiPts[i], pixel.voronoiPts[(i-1)%len(pixel.voronoiPts)])
      voronoi.add_edge(pixel.voronoiPts[i], pixel.voronoiPts[(i+1)%len(pixel.voronoiPts)])

  '''
  Takes advantage of the limited configurations of a 3x3 pixel window to estimate a generalized
  Voronoi cell for a given pixel. The Voronoi vertices are fixed to a quarter pixel length
  and are estimated based on the connections to a pixel's neighbor.
  '''
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

  '''
  Handles finding Voronoi cell vertices for the diagonal neighbors.
  '''
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

  def generateVisibleEdges(self, voronoi):
    result = set()
    for node in voronoi.nodes():
      if node not in self.VEmap:
        self.VEmap[node] = set()
      neighbors = filter(lambda x: self.dissimilarPolygons(x, node), voronoi.neighbors(node))
      for visibleEdge in self.VEmap[node]:
        neighbors = filter(lambda x: x not in visibleEdge.pts, neighbors)
      visibleEdges = [self.computeVisibleEdge(voronoi, node, neighbor) for neighbor in neighbors]

      to_remove = set()
      for i in range(len(visibleEdges)):
        for j in range(len(visibleEdges)):
          if i != j:
            ve1, ve2 = visibleEdges[i], visibleEdges[j]
            ve1 = ve1[:-1] if ve1[0] == ve1[-1] else ve1
            ve2 = ve2[:-1] if ve2[0] == ve2[-1] else ve2
            if ve1 == list(reversed(ve2)):
              to_remove.add(i)
      visibleEdges = [edge for i,edge in enumerate(visibleEdges) if i not in to_remove]

      if len(visibleEdges) == 2 and len(self.VEmap[node]) == 0:
        ve1, ve2 = visibleEdges[0], visibleEdges[1]
        hasCycle = ve1[0] == ve1[-1] and ve2[0] == ve2[-1]

        if not hasCycle:
          visibleEdges = [list(reversed(ve1))[:-1] + ve2]

      for edge in visibleEdges:
        visibleEdge = VisibleEdge(edge)
        result.add(visibleEdge)
        for point in edge:
          if point not in self.VEmap:
            self.VEmap[point] = set()
          self.VEmap[point].add(visibleEdge)

    return result

  def computeVisibleEdge(self, voronoi, node, neighbor):
    visited = set()
    visited.add(node)
    stack = [neighbor]
    result = [node]
    while len(stack) != 0:
      current = stack.pop()
      result.append(current)
      visited.add(current)

      neighbors = filter(lambda x: self.dissimilarPolygons(x, current), voronoi.neighbors(current))
      if len(neighbors) == 2:
        for neighbor in neighbors:
          if neighbor not in visited:
            stack.append(neighbor)
    return result

  def dissimilarPolygons(self, node1, node2):
    intersect = self.pixelMapping[node1] & self.pixelMapping[node2]
    if len(intersect) == 1:
      return True
    elif len(intersect) == 2:
      pixel1, pixel2 = intersect
      return self.pixelIsDissimilar(pixel1, pixel2)
    else:
      print len(intersect)

  def resolveTJunctions(self, visibleEdges):
    for point in self.VEmap:
      if len(self.VEmap[point]) == 3:
        ve0, ve1, ve2 = self.VEmap[point]
        neighbor0 = ve0[1] if ve0[0] == point else ve0[-2]
        neighbor1 = ve1[1] if ve1[0] == point else ve1[-2]
        neighbor2 = ve2[1] if ve2[0] == point else ve2[-2]
        e0, e1, e2 = map(lambda x: self.isContourEdge(point, x), [neighbor0, neighbor1, neighbor2])

        if e0 and e1 and not e2:
          self.mergeEdges(visibleEdges, point, ve0, ve1)
        elif e1 and e2 and not e0:
          self.mergeEdges(visibleEdges, point, ve1, ve2)
        elif e2 and e0 and not e1:
          self.mergeEdges(visibleEdges, point, ve2, ve0)
        else:
          a0 = self.threePointAngle(point, neighbor0, neighbor1)
          a1 = self.threePointAngle(point, neighbor1, neighbor2)
          a2 = self.threePointAngle(point, neighbor2, neighbor0)
          
          if a0 > a1 and a0 > a2:
            self.mergeEdges(visibleEdges, point, ve0, ve1)
          elif a1 > a0 and a1 > a2:
            self.mergeEdges(visibleEdges, point, ve1, ve2)
          elif a2 > a0 and a2 > a1:
            self.mergeEdges(visibleEdges, point, ve2, ve0)

  def threePointAngle(self, p0, p1, p2):
    p01 = (p0[0] - p1[0])**2 + (p0[1] - p1[1])**2
    p02 = (p0[0] - p2[0])**2 + (p0[1] - p2[1])**2
    p12 = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return math.degrees(math.acos((p01 + p02 -p12) / (2 * math.sqrt(p01 * p02))))

  def mergeEdges(self, visibleEdges, point, v1, v2):
    edge1, edge2 = v1.pts, v2.pts
    if edge1[0] != point: edge1.reverse()
    if edge2[0] != point: edge2.reverse()
    cycle1 = edge1[0] == edge1[-1]
    cycle2 = edge2[0] == edge2[-1]

    if cycle1 and cycle2:
      edge = edge1[:-1] + edge2
    elif cycle1 and not cycle2:
      edge = list(reversed(edge2))[:-1] + edge1
    else:
      edge = list(reversed(edge1))[:-1] + edge2

    mergeEdge = VisibleEdge(edge)

    visibleEdges.remove(v1)
    visibleEdges.remove(v2)

    for point in mergeEdge.pts:
      if v1 in self.VEmap[point]:
        self.VEmap[point].remove(v1)
      if v2 in self.VEmap[point]:
        self.VEmap[point].remove(v2)
      self.VEmap[point].add(mergeEdge)

    visibleEdges.add(mergeEdge)

  def isContourEdge(self, node1, node2):
    intersect = self.pixelMapping[node1] & self.pixelMapping[node2]
    if len(intersect) == 1:
      return True
    else:
      pixel1, pixel2 = intersect
      return not self.isShadingEdge(pixel1, pixel2)

  def isShadingEdge(self, pixel1, pixel2):
    dist = (pixel1.y - pixel2.y)**2 + (pixel1.u - pixel2.u)**2 + (pixel1.v - pixel2.v)**2
    return dist <= (float(100/255)**2)

  def computeSplines(self, visibleEdges):
    for edge in visibleEdges:
      spline = bspline.bspline(edge.pts, 3)

  def render(self, visibleEdges):
    drawing = svgwrite.Drawing('output.svg')
    path = []
    for edge in visibleEdges:
      path.append('M')
      path.append(self.scale(edge[0]))
      color = set([p.rgbData() for p in self.pixelMapping[edge[0]]])
      for i in range(1, len(edge.pts)-1):
        path.append('Q')
        path.append(self.scale(edge[i]))
        path.append(self.scale(edge[i+1]))
        color &= set([p.rgbData() for p in self.pixelMapping[edge[i]]]) & set([p.rgbData() for p in self.pixelMapping[edge[i+1]]])
      #path.append('Z')
    drawing.add(drawing.path(path, stroke=svgwrite.rgb(0,0,0), fill=svgwrite.rgb(255,255,255)))
    drawing.save()

  def scale(self, point):
    return (int(point[0] * 100), int(point[1] * 100))

'''
Main method.
'''
if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'Missing argument: \'python depixelize.py <target_image>\''
    exit()
  depixelizer = Depixelizer()
  depixelizer.depixelize(sys.argv[1])
