import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from astropy import constants as const
from astropy import units as u

# Create initial plot with adjusted figure size
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.1, right=0.8)  # Adjust left and right margins

# Define the function that generates the plot
def update_plot(val):
    q = slider1.val
    R = slider2.val
    a_array = np.linspace(2.1, 3.3, 100)  # Adjust the endpoint of the x-axis
    Egg = ( 0.49*q**(2/3) ) / ( 0.6*q**(2/3) + np.log(1+q**(1/3)) )
    Rd = R * const.R_sun.cgs.value * u.cm.to(u.au)
    f_array = Rd / a_array / Egg
    
    ax.clear()
    ax.plot(a_array, f_array)
    ax.set_title('filling ratio')
    ax.set_xlabel('a [AU]')
    ax.set_ylabel('filling ratio')
    plt.draw()

# Define sliders
axcolor = 'lightgoldenrodyellow'
ax_param1 = plt.axes([0.85, 0.2, 0.03, 0.65], facecolor=axcolor)
ax_param2 = plt.axes([0.91, 0.2, 0.03, 0.65], facecolor=axcolor)

slider1 = Slider(ax_param1, 'q', 1, 100, valinit=10, orientation='vertical')
slider2 = Slider(ax_param2, 'R [R_sun]', 210, 323, valinit=250, orientation='vertical')

# Attach the update function to the sliders
slider1.on_changed(update_plot)
slider2.on_changed(update_plot)

# Define the reset button
reset_button_ax = plt.axes([0.8, 0.9, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axcolor, hovercolor='0.975')

# Define the reset function
def reset(event):
    slider1.reset()
    slider2.reset()

reset_button.on_clicked(reset)

# Display the plot and sliders
plt.show()
