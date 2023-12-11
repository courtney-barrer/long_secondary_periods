import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Slider
from astropy import constants as const
from astropy import units as u

# Define the function that generates the values for the plot
def update_plot(q, R, param):
    Egg = ( 0.49*q**(2/3) ) / ( 0.6*q**(2/3) + np.log(1+q**(1/3)) )
    Rd = R * const.R_sun.cgs.value * u.cm.to(u.au)
    f = Rd / param / Egg
    return f

# Initial values
Period = 757   # in days
Omega = 2 * np.pi / (Period*86400)   # in rad/s
initial_param = 3.0
initial_central_value = 3.0
x_values = np.linspace(1, 100, 500)
y_values = np.linspace(150, 350, 100)
X, Y = np.meshgrid(x_values, y_values)
Z = update_plot(X, Y, initial_param)

# Create the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Adjust the bottom to make room for the slider

# Plot the initial surface with a divergent colormap
cmap = plt.get_cmap('coolwarm')
divnorm = colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=1.5)
plot = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=divnorm)
contour = ax.contour(X, Y, Z, levels=[1], colors='k', linestyles='solid')
ax.set(xlabel='q', ylabel='radius (solar radii)')
plt.title('The line shows f=1')

# Add colorbar
cbar = plt.colorbar(plot, ax=ax)
cbar.set_label('filling factor (f)')

# Add a slider for parameter adjustment
ax_param = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
param_slider = Slider(ax_param, 'a [AU]', 2.1, 3.5, valinit=initial_param)

# Function to update the plot when the slider is moved
def update(val):
    global contour
    param = param_slider.val
    Z = update_plot(X, Y, param)
    plot.set_array(Z.flatten())
    # Update the contour line
    if contour is not None:
        for coll in contour.collections:
            coll.remove()
    contour = ax.contour(X, Y, Z, levels=[1], colors='k', linestyles='solid')
    fig.canvas.draw_idle()

# Attach the slider's update function to the slider
param_slider.on_changed(update)

plt.show()
