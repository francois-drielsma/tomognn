from copy import deepcopy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot

def draw2d(pixel_sets, pixel_mode='square'):
    '''
    Draws a set of pixel plots side by side

    Args:
        pixel_sets (list(dict)): List of sparse sets of pixels
        pixel_mode (str)       : Pixel drawing method
    '''
    # Make subplots
    if not isinstance(pixel_sets, list):
        pixel_sets = [pixel_sets]
    fig = make_subplots(rows=1, cols=len(pixel_sets))

    # Add the graphs together
    ranges = []
    if pixel_sets is not None:
        for i, ps in enumerate(pixel_sets):
            fig.add_trace(pixel_graph(ps, pixel_mode)[0], row=1, col=i+1)
            ranges.append(ps.ranges)

    # Draw
    fig.update_layout(layout2d(ranges))
    iplot(fig)

def draw3d(end_points=None, points=None, voxel_set=None, voxel_mode='cube',
           pixel_sets=None, bases=None, edge_index=None, ranges=None):
    '''
    Draws a combination of lines, voxels and points upon request

    Args:
        end_points (list(array)): List of start and end coordinates of lines
        points (array)          : Matrix of 3d points (x,y,z,val)
        voxel_set (dict)        : Sparse set of voxels
        voxel_mode (str)        : Voxel drawing method
        pixel_sets (list(dict)) : List of sparse sets of pixels
        bases (list(array))     : List of projection axis pairs
        edge_index (array)      : Array of edges between projected points [P_i,P_j,i,j]
        ranges (array)          : Boundaries of the volume the objects live in
    '''
    # Add the graphs together
    graphs = []
    if end_points is not None:
        graphs += line_graph(end_points)
    if points is not None:
        graphs += point_graph(points)
    if voxel_set is not None:
        graphs += voxel_graph(voxel_set, voxel_mode)
        if ranges is None:
            ranges = voxel_set.ranges
    if edge_index is not None:
        assert pixel_sets is not None and bases is not None and len(pixel_sets) == len(bases)
        graphs += edge_index_graph(edge_index, pixel_sets, bases, ranges)
    if pixel_sets is not None:
        assert len(pixel_sets) == len(bases), 'Provide projection bases if projections are drawn'
        graphs += boundary_graph(ranges)
        new_ranges = deepcopy(ranges)
        for i in range(len(pixel_sets)):
            graphs += projection_graph(pixel_sets[i], bases[i], ranges, new_ranges)
        ranges = new_ranges
    assert len(graphs), "Need to pass one of end_points, points or voxel_set"

    # Draw
    assert ranges is not None, "If no voxel set is specified, please provide the volume boundaries"
    fig = go.Figure(graphs, layout3d(ranges))
    iplot(fig)

def line_graph(end_points):
    '''
    Returns a graph of lines, provided their end points (starts, ends)

    Args:
        end_points (list(array)): List of start and end coordinates
    '''
    # Alternate start and ends, pad with None between disjoint lines
    points = np.hstack((*end_points, np.full(end_points[0].shape, None))).reshape(-1,3)

    # Initialize graph object
    graph = go.Scatter3d(x = points[:,0],
                         y = points[:,1],
                         z = points[:,2],
                         line = dict(color = 'blue', width = 2))

    return [graph]

def point_graph(points):
    '''
    Returns a graph of points

    Args:
        points (array): Matrix of 3d points (x,y,z,val)
    '''
    # Initialize graph object
    graph = go.Scatter3d(x = points[:,0],
                         y = points[:,1],
                         z = points[:,2],
                         mode = 'markers',
                         marker = dict(color = points[:,-1], size=3),
                         hovertext = points[:,-1])

    return [graph]

def pixel_graph(pixel_set, mode='square'):
    '''
    Returns a graph of pixels (represented as bins or centroid points)

    Args:
        pixel_set (dict): Sparse set of pixels
        mode (str)      : Pixel drawing method
    '''
    # Initialize voxel graph (one cuboid per voxel)
    if mode == 'square':
        centers = pixel_set.centers
        values  = pixel_set.values
        graph = go.Histogram2d(x = centers[:,0],
                               y = centers[:,1],
                               z = values,
                               xbins = dict(
                                   start = pixel_set.ranges[0,0],
                                   end   = pixel_set.ranges[0,1],
                                   size  = pixel_set.bin_size[0]
                               ),
                               ybins = dict(
                                   start = pixel_set.ranges[1,0],
                                   end   = pixel_set.ranges[1,1],
                                   size  = pixel_set.bin_size[1]
                               ),
                               histfunc = 'sum',
                               coloraxis = 'coloraxis'
                              )

        return [graph]
    elif mode == 'marker':
        raise NotImplementedError('TODO')
    else:
        raise ValueError('Voxel drawing mode not in [\'square\', \'marker\']')

def voxel_graph(voxel_set, mode='cube'):
    '''
    Returns a graph of voxels (represented as cuboids or centroid points)

    Args:
        voxel_set (dict): Sparse set of voxels
        mode (str)      : Voxel drawing method
    '''
    # Initialize voxel graph (one cuboid per voxel)
    if mode == 'cube':
        graphs  = []
        lcoords = voxel_set.lower_limits
        values  = voxel_set.values
        min_val, max_val = min(voxel_set.values), max(voxel_set.values)

        for i in range(voxel_set.size):
            color = get_object_color(min_val, max_val, voxel_set.values[i], 'Inferno')
            graphs += cuboid_graph(*lcoords[i], *voxel_set.bin_size, color=color, hovertext=values[i])

        return graphs
    elif mode == 'marker':
        return point_graph(np.hstack((voxel_set.centers, voxel_set.values[:,None])))
    else:
        raise ValueError('Voxel drawing mode not in [\'cube\', \'marker\']')

def projection_graph(pixel_set, base, ranges, new_ranges):
    '''
    Returns a graph of projected points and their associated projecion plane

    Args:
        pixel_set (dict)  : Sparse sets of pixels
        base (array)      : Projection axis pair
        ranges (array)    : Boundaries of the volume the objects live in
        ranges_new (array): Boundaries of the volume to include the projections
    '''
    # Draw the projection plane
    normal   = np.cross(*base)
    centroid = np.mean(pixel_set.vertices, axis=0) @ base
    offset   = np.sqrt(2)-np.dot(centroid - np.mean(ranges, axis=1), normal)
    vertices = pixel_set.vertices @ base + offset*normal
    graphs   = [go.Mesh3d(x = vertices[:,0],
                          y = vertices[:,1],
                          z = vertices[:,2],
                          opacity = 0.25,
                          color = 'lightblue')]

    # Extend range to accomodate projection plane
    plane_ranges = np.vstack((np.min(vertices, axis=0), np.max(vertices, axis=0))).T
    new_ranges[:,0]  = np.min(np.vstack((plane_ranges[:,0], new_ranges[:,0])), axis=0)
    new_ranges[:,1]  = np.max(np.vstack((plane_ranges[:,1], new_ranges[:,1])), axis=0)

    # Draw the intercepts of the projection rays with the projection planes and the plane normals
    intercepts = pixel_set.centers @ base + offset*normal
    graphs += point_graph(np.hstack((intercepts, pixel_set.values.reshape(-1,1))))

    return graphs

def edge_index_graph(edge_index, pixel_sets, bases, ranges):
    '''
    Returns a set of lines corresponding to edges joining projection points

    Args:
        edge_index (array)     : Array of edges between projected points [P_i,P_j,i,j]
        pixel_sets (list(dict)): List of sparse sets of pixels
        bases (list(array))    : List of projection axis pairs
        ranges (array)         : Boundaries of the volume the objects live in
    '''
    # Get the projected points
    intercepts = []
    for i in range(len(pixel_sets)):
        normal   = np.cross(*bases[i])
        centroid = np.mean(pixel_sets[i].vertices, axis=0) @ bases[i]
        offset   = np.sqrt(2)-np.dot(centroid - np.mean(ranges, axis=1), normal)
        intercepts.append(pixel_sets[i].centers @ bases[i] + offset*normal)

    # Use the points as edge vertices
    edge_vertices = np.empty((len(edge_index)*3, 3), dtype=np.float64)
    for k, e in enumerate(edge_index):
        edge_vertices[3*k] = intercepts[e[0]][e[2]]
        edge_vertices[3*k+1] = intercepts[e[1]][e[3]]
        edge_vertices[3*k+2] = [None,None,None]

    graph = go.Scatter3d(x = edge_vertices[:,0],
                         y = edge_vertices[:,1],
                         z = edge_vertices[:,2],
                         mode = 'lines',
                         line = dict(
                            color='gray',
                            width=1))

    return [graph]

def boundary_graph(ranges):
    '''
    Returns a cuboid graph which encompasses the whole volume

    Args:
        ranges (array): Boundaries of the volume the objects live in
    '''
    llims, dimensions = ranges[:,0], ranges[:,1]-ranges[:,0]
    return cuboid_graph(*llims, *dimensions, color='gray', opacity=0.1)

def cuboid_graph(lx, ly, lz, dx, dy=None, dz=None, color=None, hovertext='', opacity=0.5):
    '''
    Define a cuboid graph in plotly

    Args:
        lx (double): Lower x value
        ly (double): Lower y value
        lz (double): Lower z value
        dx (double): Extent in x
        dy (double): Extent in y
        dz (double): Extent in z
    Returns:
        plotly.graph_objs.Mesh3d: Plotly graph object of a cube
    '''
    if not dy: dy = dx
    if not dz: dz = dx
    if not color: color = '#DC143C'
    return [go.Mesh3d(
        # 8 vertices of a cube
        x = lx + dx*np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        y = ly + dy*np.array([0, 1, 1, 0, 0, 1, 1, 0]),
        z = lz + dz*np.array([0, 0, 0, 0, 1, 1, 1, 1]),

        i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity = opacity,
        color = color,
        hovertext = hovertext,
        flatshading = True
    )]

def get_object_color(min, max, val, colorscale):
    """
    Get color given the value of an object and a plotly colorscale
    (if multiple objects are drawn, their colors is arbitrary and does
    not follow the color value given to the object)

    Args:
        min (double)               : Minimum value of the range of values to be drawn
        max (double)               : Maxmimum value of the range of values to be drawn
        val (double)               : Value of the object
        colorscale (string or list): Plotly colorscale (either name of user defined list)
    Returns:
        str: Plotly color
    """
    # If the colorscale is a string, look for it in plotly express
    if isinstance(colorscale, str):
        import plotly.express as px
        colorscale = getattr(px.colors.sequential, colorscale)
        colorscale = [[i/(len(colorscale)-1), c] for i, c in enumerate(colorscale)]

    # Get the value adjusted to the value range
    if (max-min) > 0:
        frac_val = (val-min)/(max-min)
    else:
        frac_val = 0.5

    # Find the color ID
    if frac_val == 0:
        color_id = 0
    elif frac_val == 1:
        color_id = len(colorscale)-1
    else:
        cs_limits = [color[0] for color in colorscale]
        color_id  = np.where(cs_limits/frac_val > 1)[0][0]-1

    return colorscale[color_id][1]

def layout2d(ranges, **kwargs):
    """
    Produces Plotly layout object for a box

    Args:
        ranges (array): Boundaries of the volume the objects live in
    Returns:
        plotly.graph_objs.Layout: 3d layout
    """
    # Initialize the base layout
    layout = go.Layout(
        height = 320,
        width  = len(ranges)*320,
        showlegend = False,
        margin = dict(l = 25, r = 25, b = 25, t = 25),
        coloraxis = dict(
            colorscale='Inferno',
            colorbar=dict(outlinecolor = 'black', outlinewidth=2)
        ),
        **kwargs
    )

    # Don't show the empty bins
    layout['coloraxis']['colorscale'] = [(1e-9, '#000004')] + list(layout['coloraxis']['colorscale'][1:])
    layout['coloraxis']['colorscale'] = [(0.0, '#FFFFFF')] + list(layout['coloraxis']['colorscale'])

    # Update the axes properties for each of the plots
    for i, r in enumerate(ranges):
        layout.update({
            f'xaxis{i+1}': go.layout.XAxis(
                nticks = 10, range = ranges[i][0], showticklabels=True,
                title=r'$x_{%d0}$'%i, title_standoff=0,
                mirror=True, showline=True, ticks='outside', linecolor='black', linewidth=2
            ),
            f'yaxis{i+1}': go.layout.YAxis(
                nticks = 10, range = ranges[i][1], showticklabels=True,
                title=r'$x_{%d1}$'%i, title_standoff=0,
                mirror=True, showline=True, ticks='outside', linecolor='black', linewidth=2
            )
        })

    return layout

def layout3d(ranges, titles=None, **kwargs):
    """
    Produces Plotly layout object for a box

    Args:
        ranges (array): Boundaries of the volume the objects live in
    Returns:
        plotly.graph_objs.Layout: 3d layout
    """
    dimensions = ranges[:,1]-ranges[:,0]
    layout = go.Layout(
        showlegend = False,
        width  = 500,
        height = 500,
        margin = dict(l=0, r=0, b=0, t=0),
        scene  = dict(
            xaxis = dict(nticks=10, range = ranges[0], showticklabels=True,
                         title='x' if titles is None else titles[0],
                         backgroundcolor="white", gridcolor="lightgray",
                         showbackground=True,
                        ),
            yaxis = dict(nticks=10, range = ranges[1], showticklabels=True,
                         title='y' if titles is None else titles[1],
                         backgroundcolor="white", gridcolor="lightgray",
                         showbackground=True
                        ),
            zaxis = dict(nticks=10, range = ranges[2], showticklabels=True,
                         title='z' if titles is None else titles[2],
                         backgroundcolor="white", gridcolor="lightgray",
                         showbackground=True,
                        ),
            aspectmode = 'manual',
            aspectratio = dict(x=dimensions[0], y=dimensions[1], z=dimensions[2]),
            camera = dict(
                up = dict(x=0, y=0, z=1),
                center = dict(x=0, y=0, z=-0.1),
                eye = dict(x=1.45, y=1.45, z=0.1)
            ),
        ),
        **kwargs
    )
    return layout
