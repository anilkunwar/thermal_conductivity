import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def sph2cart(r, phi, tta):
    ''' r is from 0 to infinity '''
    ''' phi is from 0 to 2*pi '''
    ''' tta is from 0 to pi '''
    x = r * np.sin(tta) * np.cos(phi)
    y = r * np.sin(tta) * np.sin(phi)
    z = r * np.cos(tta)
    return x, y, z

# Get user input for thermal conductivity values
k11 = st.slider("k11", 0.0, 100.0, 20.00, 0.01)
k22 = st.slider("k22", 0.0, 100.0, 20.00, 0.01)
k33 = st.slider("k33", 0.0, 100.0, 40.00, 0.01)

# phi running from 0 to pi and tta from 0 to pi
phi = np.linspace(0, 2 * np.pi, 100)
tta = np.linspace(0, np.pi, 100)
# meshgrid to generate points
phi, tta = np.meshgrid(phi, tta)

# thermal conductivity tensor
K = np.array([[k11, 0, 0], [0, k22, 0], [0, 0, k33]])

# matrix of cartesian coordinates
X, Y, Z = sph2cart(1, phi, tta)
XYZ = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

# calculate conductivity values at each point
Kvals = []
for i in range(XYZ.shape[0]):
    k = XYZ[i] @ K @ XYZ[i].T
    Kvals.append(k)
Kvals = np.array(Kvals).reshape(X.shape)

# create the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# calculate the radii in each axis based on thermal conductivity values
radius_a = np.sqrt(Kvals / k11)
radius_b = np.sqrt(Kvals / k22)
radius_c = np.sqrt(Kvals / k33)

# scale the coordinates with the radii
X_scaled = X * radius_a
Y_scaled = Y * radius_b
Z_scaled = Z * radius_c

# set the colormap and normalization
#cmap = cm.get_cmap('RdBu')
cmap = cm.get_cmap('hsv')
vmin = np.min(Kvals)
vmax = np.max(Kvals)
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# plot the surface with scaled coordinates
surf = ax.plot_surface(X_scaled, Y_scaled, Z_scaled,
                       rstride=1, cstride=1, facecolors=cmap(norm(Kvals)), linewidth=0.1, alpha=1.0)

# add the colorbar
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label('Thermal Conductivity (W/m K)',  fontsize=32)
# adjust the font size of colorbar tick labels
cbar.ax.tick_params(labelsize=25)

# set the axis limits
lim = np.max(np.abs([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]))
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)

# set the axis labels
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('c')

# set the title
ax.set_title('Anisotropic Thermal Conductivity Tensor', fontsize = 24)
# Increase font size of tick labels
ax.tick_params(labelsize=22)

# Customize figure size
fig.set_size_inches(10, 10)

# save the figure with increased width and height
#fig.savefig('thermal_conductivity.png', dpi=300, bbox_inches='tight')

# display the figure in Streamlit
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html,width = 1000, height=1200)
#components.html(fig_html,width = 1000, height=6000)

