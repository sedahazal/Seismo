from bokeh.models import PanTool, ResetTool, HoverTool, WheelZoomTool
from bokeh.io import output_file, show,save
from bokeh.plotting import figure
import os
def graph_creater(output_path,file_name, x, y, x_label, y_label, width, height, logx="auto"):
    file_path = os.path.join(output_path,file_name)
    if file_name in os.listdir(output_path):
        os.remove(file_path)
    output_file(file_path)
    f = figure(tools=[PanTool(), ResetTool(), WheelZoomTool()], x_axis_type=logx)
    hover = HoverTool(tooltips=[("x", "$x"), ("y", "$y")])

    f.add_tools(hover)
    f.toolbar_location = 'above'
    f.toolbar.logo = None

    f.plot_width = width
    f.plot_height = height
    f.xaxis.axis_label = x_label
    f.yaxis.axis_label = y_label

    f.line(x, y)

    #show(f)
    save(f)