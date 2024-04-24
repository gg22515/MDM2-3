import rdata
import numpy as np
from manim import *

parsed = rdata.parser.parse_file("5FishTest.RData")
converted = rdata.conversion.convert(parsed)
data = converted["data"]

x2Frames = data.x2
y2Frames = data.y2

x1Frames = data.x1
y1Frames = data.y1

x3Frames = data.x3
y3Frames = data.y3

x4Frames = data.x4
y4Frames = data.y4

x5Frames = data.x5
y5Frames = data.y5

xBounds = [min(data.x1.min(),data.x2.min(),data.x3.min(),data.x4.min(),data.x5.min())
           ,max(data.x1.max(),data.x2.max(),data.x3.max(),data.x3.max(),data.x5.max())]
yBounds = [min(data.y1.min(),data.y2.min(),data.y3.min(),data.y4.min(),data.y5.min()),
           max(data.y1.max(),data.y2.max(),data.y3.max(),data.y4.max(),data.y5.max())]

class TestScene(Scene):
  def construct(self):

    number_plane = NumberPlane(
        background_line_style={
            "stroke_color": TEAL,
            "stroke_width": 4,
            "stroke_opacity": 0.6
        },
        x_range = [-50,50],
        y_range = [-50,50]
    )

    text = Text("Fish Movement").scale(2)
    circle = Circle()
    circle.set_fill(RED, opacity = 0.7)

    triangle = Triangle()

    square = Square()
    square.rotate(PI/4)

    circle2 = Circle()

    self.play(Create(circle))
    self.play(Transform(circle, square))

    square2 = Square()
    square2.next_to(circle, RIGHT, buff = 0.5)

    self.play(Create(square2))

    self.play(circle.animate.rotate(PI/4))

    self.play(square2.animate.rotate(PI/4))

    self.play(Transform(circle, circle2))

    self.play(Transform(square2, circle))

    self.remove(square2)
    
    self.play(Transform(circle, triangle))

    self.add(number_plane)

    self.play(number_plane.animate.rotate(PI/6), circle.animate.rotate(PI/6))

    self.add(text)

    self.play(Create(text))

    #self.wait()

    self.remove(text, circle, number_plane)

    axes = Axes(x_range=[data.x1.min(), data.x1.max(), 5], y_range=[data.y1.min(), data.y1.max(), 5])
    self.add(axes)
    self.play(Create(axes))


    x = list(np.array(x1Frames))

    y = list(np.array(y1Frames))

    for index in range(5):
      dot = Dot(axes.c2p(x[index], y[index]), color = BLUE)
      self.play(Create(dot))

    #line = axes.plot_line_graph(x,y, add_vertex_dots = True)

    #self.play(Create(line))

    self.wait()
