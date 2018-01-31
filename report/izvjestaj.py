##
from pyx import *

g = graph.graphxy(width=8)
g.plot(graph.data.function("y(x)=sin(x)/x", min=-15, max=15))
g.writePDFfile("function")
print r'\includegraphics{function}'

