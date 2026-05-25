import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row
from bokeh.models import Slider, ColumnDataSource
from bokeh.palettes import Category10

# Fixed range for x and y
x_vals = np.linspace(0, 2, 200)
y_vals = np.linspace(0, 2, 200)
X, Y = np.meshgrid(x_vals, y_vals)

threshold = 0.02  # zero contour tolerance

def compute_zero_points(a, b, c):
    J11 = 1 - 2*X - b*Y       # depends on b only for J11
    J12 = -b * X              # depends on b only for J12
    # a and c don't affect these partial derivatives, but included as params

    idx_j11 = np.abs(J11) < threshold
    idx_j12 = np.abs(J12) < threshold

    return {
        'j11_x': X[idx_j11],
        'j11_y': Y[idx_j11],
        'j12_x': X[idx_j12],
        'j12_y': Y[idx_j12]
    }

# Initial parameter values
a_init = 0.3
b_init = 0.5
c_init = 0.1

data = compute_zero_points(a_init, b_init, c_init)

source_j11 = ColumnDataSource(data=dict(x=data['j11_x'], y=data['j11_y']))
source_j12 = ColumnDataSource(data=dict(x=data['j12_x'], y=data['j12_y']))

p = figure(width=600, height=600,
           title=f"Zero Contours of Jacobian Entries\n(a={a_init:.2f}, b={b_init:.2f}, c={c_init:.2f})",
           x_axis_label='x', y_axis_label='y',
           tools="pan,wheel_zoom,reset")

p.scatter('x', 'y', source=source_j11, size=6, color=Category10[10][0], alpha=0.6, legend_label="J11=0")
p.scatter('x', 'y', source=source_j12, size=6, color=Category10[10][1], alpha=0.6, legend_label="J12=0")

p.legend.location = "top_right"
p.legend.click_policy = "hide"

# Sliders for parameters
slider_a = Slider(start=0, end=1, value=a_init, step=0.01, title="a (stimulus to y)")
slider_b = Slider(start=0, end=2, value=b_init, step=0.01, title="b (interaction strength)")
slider_c = Slider(start=0, end=1, value=c_init, step=0.01, title="c (decay rate)")

def update(attr, old, new):
    a = slider_a.value
    b = slider_b.value
    c = slider_c.value
    new_data = compute_zero_points(a, b, c)
    source_j11.data = dict(x=new_data['j11_x'], y=new_data['j11_y'])
    source_j12.data = dict(x=new_data['j12_x'], y=new_data['j12_y'])
    p.title.text = f"Zero Contours of Jacobian Entries\n(a={a:.2f}, b={b:.2f}, c={c:.2f})"

for slider in (slider_a, slider_b, slider_c):
    slider.on_change('value', update)

layout = column(row(slider_a, slider_b, slider_c), p)
curdoc().add_root(layout)
curdoc().title = "Jacobian Zero Contours Interactive"
