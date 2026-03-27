# Script to generate the mesh file for a 3D, rectangular mesh and quarter-spherical charge with sensor and tracer elements for automatic simulation termination
# Assumptions: mesh's bottom left corner is at the origin; quarter-spherical (octant) charge centered at origin;
# part ID of non-explosive is 1 and ID of explosive is 2, region is rectangular
# Haena Lee, Sept 2025

import numpy as np
import os
import sys

# -----------------------------
# Model parameters
# -----------------------------
OUTPUT_FILENAME  = "mesh.inc"
ELEMENT_SIZE_CM = 0.4                       # element size in cm
EXPL_RADIUS_CM  = 16.0                      # radius of spherical explosive charge in cm
REGION_X_DIM    = 36                        # x dimension in cm of the entire 3D region
REGION_Y_DIM    = 36
REGION_Z_DIM    = 36
FIXED_COORDS     = [(0.0, 0.0, 0.0)]        # list of fixed coordinates (x,y,z)
SENSOR_OFFSET_EL = 1                        # number of elements to offset sensor planes from max boundaries
SENSOR_SET_ID    = 9001                     # sensor element set ID
TRACER_ELSET_ID  = 7001                     # tracer element set ID
TRACER_NSET_ID   = 7101                     # tracer node set ID


# -----------------------------
# Make sure there are an integer number of elements
# -----------------------------
def check_integer_num_elements(element_size, outer_dims):
    xf, yf, zf = outer_dims
    nx = xf / element_size
    ny = yf / element_size
    nz = zf / element_size
    tol = 1e-10

    def is_int(x):
        return abs(x - round(x)) <= tol

    if not (is_int(nx) and is_int(ny) and is_int(nz)):
        print(
            f"\nOuter dims are not divisible by element_size.\n"
            f"There are {nx:.2f} elements in x, {ny:.2f} in y, and {nz:.2f} in z.\n"
            "Please choose a different element size or outer dimensions.\n"
        )
        return False
    return True


# -----------------------------
# Helper functions
# -----------------------------
# Generate node IDs and coordinates; NOT ordered in the same manner as in `fine.inc`
def generate_nodes(element_size, outer_dims):
    xf, yf, zf = outer_dims
    tol = 1e-10

    #Check that there are an integer # of elements in the entire region
    nx_expl = xf / element_size
    ny_expl = yf / element_size
    nz_expl = zf / element_size

    def is_int(x):  #check if x is an integer within tolerance
        return abs(x - round(x)) <= tol

    if not (is_int(nx_expl) and is_int(ny_expl) and is_int(nz_expl)):   #if there's not an integer # of elements
        print(
            f"There are {nx_expl:.2f} elements in x, {ny_expl:.2f} in y, and {nz_expl:.2f} in z.\n"
            "Please choose a different element size or outer dimensions."
        )
        return None

    #Convert all lengths to indices, rounding to nearest int
    def to_index(L):
        return int(round(L / element_size))

    nxf = to_index(xf)
    nyf = to_index(yf)
    nzf = to_index(zf)

    #generate nodes for the entire region, in increasing x, y, z order
    nodes = []
    for k in range(nzf + 1):
        for j in range(nyf + 1):
            for i in range(nxf + 1):
                x = i * element_size
                y = j * element_size
                z = k * element_size
                nodes.append((x, y, z))

    nodes = np.array(nodes, dtype=float)
    node_IDs = np.arange(1, nodes.shape[0] + 1).reshape(-1, 1)    #generate column of node IDs
    nodes = np.hstack((node_IDs, nodes))
    return nodes


#Add translational and rotational constraints to nodes
def add_constraints(nodes, fixed_coords):
    coords = nodes[:, 1:4]
    tc = np.full((coords.shape[0],), 0)   #unconstrained by default
    rc = np.full((coords.shape[0],), 7)   #constrain all xyz rotations

    for i, (x, y, z) in enumerate(coords):
        if np.isclose(x, 0.0):   #yz plane
            tc[i] = 1            #constrain x displacement
        if np.isclose(y, 0.0):   #xz plane
            tc[i] = 2            #constrain y displacement
        if np.isclose(z, 0.0):   #xy plane
            tc[i] = 3            #constrain z displacement

        if np.isclose(y, 0.0) and np.isclose(z, 0.0):   #x edge at origin
            tc[i] = 5            #constrain y and z
        if np.isclose(x, 0.0) and np.isclose(z, 0.0):   #y edge at origin
            tc[i] = 6            #constrain x and z
        if np.isclose(x, 0.0) and np.isclose(y, 0.0):   #z edge at origin
            tc[i] = 4            #constrain x and y

        for fx, fy, fz in fixed_coords:
            if np.isclose([x, y, z], [fx, fy, fz]).all():
                tc[i] = 7        #fix specified coordinates
                break

    #Add tc and rc columns
    constraints = np.column_stack((tc, rc))
    nodes = np.hstack((nodes, constraints))
    return nodes


# Generate the hexahedral elements; NOT ordered in the same manner as in `fine.inc`
def generate_elements(nodes, element_size, outer_dims, expl_radius):
    xf, yf, zf = outer_dims
    part_nonexpl = 1        #part ID of the non-explosive region
    part_expl = 2
    elements = []

    #Create node ID map with columns: node ID, x, y, z
    nodes = np.asarray(nodes)
    ids = nodes[:, 0].astype(int)
    xs = nodes[:, 1].astype(float)
    ys = nodes[:, 2].astype(float)
    zs = nodes[:, 3].astype(float)

    #Convert physical coordinates to integer grid indices (i,j,k) based on element size
    scale = 1.0 / float(element_size)   #multiply by this instead of dividing to prevent floating pt errors
    tol = 1e-10
    ix = np.rint(xs * scale + tol).astype(int)
    iy = np.rint(ys * scale + tol).astype(int)
    iz = np.rint(zs * scale + tol).astype(int)
    indices = np.stack([ix, iy, iz], axis=1)

    #Dictionary to get node ID from integer indices
    index_to_id = {}
    for idx, nid_val in zip(indices, ids):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        index_to_id[key] = int(nid_val)

    def nid(i, j, k):
        return index_to_id[(i, j, k)]   #corresponding node ID for the integer indices

    #Convert lengths to grid indices, rounding to nearest int
    #E.g., if xf = 100 and element_size = 0.5, to_index(100) -> 200
    def to_index(L):
        return int(round(L / element_size))

    nxf = to_index(xf)
    nyf = to_index(yf)
    nzf = to_index(zf)

    #Define the 8 vertices of each element and return their corresponding node IDs
    def element_node_IDs(i, j, k):
        try:     #try finding the node ID for each vertex
            return [
                nid(i, j, k),       nid(i + 1, j, k),       nid(i + 1, j + 1, k),       nid(i, j + 1, k),
                nid(i, j, k + 1),   nid(i + 1, j, k + 1),   nid(i + 1, j + 1, k + 1),   nid(i, j + 1, k + 1)
            ]
        except KeyError:
            return None

    #if the x and y coordinates of the midpoint of the element are within expl_radius, set element as explosive element
    for k in range(nzf):
        for j in range(nyf):
            for i in range(nxf):
                ns = element_node_IDs(i, j, k)
                if ns is None:
                    continue
                #get coords of lower left corner and add half the element size to get midpoint coords
                x_mid = (i + 0.5) * element_size
                y_mid = (j + 0.5) * element_size
                z_mid = (k + 0.5) * element_size
                dist_sq = x_mid**2 + y_mid**2 + z_mid**2
                part = part_expl if dist_sq <= (expl_radius + tol)**2 else part_nonexpl
                elements.append([part, *ns])

    elements = np.asarray(elements, dtype=int)
    element_IDs = np.arange(1, elements.shape[0] + 1, dtype=int).reshape(-1, 1)    #generate column of element IDs
    elements = np.hstack((element_IDs, elements))
    return elements


#Return the sensor elements offset from the top and right boundaries by the # of elements specified by sensor_offset; assuming region's thickness is z=1 
def define_sensor_elements(node_section, element_section, element_size, outer_dims, sensor_offset):
    xf, yf, zf = map(float, outer_dims)
    es = float(element_size)
    nxf = int(round(xf / es))
    nyf = int(round(yf / es))
    nzf = int(round(zf / es))

    #x and y coordinates of the top and right edges of the sensor elements 
    sensor_x_coord = (nxf - sensor_offset) * es
    sensor_y_coord = (nyf - sensor_offset) * es
    sensor_z_coord = (nzf - sensor_offset) * es

    #dictionary to get coordinates from node IDs 
    nodeID_to_coords = {} 
    for row in node_section: 
        node_id = int(row[0]) 
        nodeID_to_coords[node_id] = tuple(map(float, row[1:4])) 
    
    sensor_elements = set()     #use set to avoid duplicate elements (e.g., the one where the row and column intersect) 
    tol = 1e-10

    for row in element_section:
        eid = int(row[0])
        nids = [int(v) for v in row[2:10]]
        x_coords = [nodeID_to_coords[n][0] for n in nids]
        y_coords = [nodeID_to_coords[n][1] for n in nids]
        z_coords = [nodeID_to_coords[n][2] for n in nids]

        on_sensor_x = abs(max(x_coords) - sensor_x_coord) <= tol
        on_sensor_y = abs(max(y_coords) - sensor_y_coord) <= tol
        on_sensor_z = abs(max(z_coords) - sensor_z_coord) <= tol

        if on_sensor_x or on_sensor_y or on_sensor_z:
            sensor_elements.add(eid)

    return sensor_elements


#Return the tracer elements along the bottom boundary and lowest z layer (y=0, z=0)
def define_tracer_elements(node_section, element_section):
    tracer_elements = set()
    tol = 1e-10

    #Dictionary to get coordinates from node IDs
    nodeID_to_coords = {}
    for row in node_section:
        node_id = int(row[0])
        nodeID_to_coords[node_id] = tuple(map(float, row[1:4]))

    for row in element_section:
        eid = int(row[0])
        nids = [int(r) for r in row[2:10]]
        y_coords = [nodeID_to_coords[n][1] for n in nids]
        z_coords = [nodeID_to_coords[n][2] for n in nids]
        if abs(min(y_coords)) <= tol and abs(min(z_coords)) <= tol:
            tracer_elements.add(eid)

    return tracer_elements


#Return the nodes of the tracer elements
def define_tracer_nodes(element_section, tracer_elements):
    tracer_nodes = set()
    bottom_face_indices = (0, 1, 2, 3)      #the 4 node IDs that make up the bottom face of each element

    #Dictionary to get node IDs from element ID
    elemID_to_nodeIDs = {}
    for row in element_section:
        nids = [int(v) for v in row[2:10]]
        elemID_to_nodeIDs[int(row[0])] = nids

    for eid in tracer_elements:
        nids = elemID_to_nodeIDs[eid]
        for i in bottom_face_indices:
            tracer_nodes.add(nids[i])

    return tracer_nodes


#Format the node and element sections into the output file, in the same manner as 'fine.inc'
def format_sections_into_file(node_section, element_section, sensor_elements, sensor_set_id,
    tracer_elements, tracer_elset_id, tracer_nodes, tracer_nset_id, output_path):
    #Write integers in 10 character wide columns, 10 values per line
    def write_set_section(file, values, per_line=8):
        line = []
        for i, v in enumerate(sorted(values), 1):
            line.append(f"{int(v):10d}")
            if i % per_line == 0:
                file.write("".join(line) + "\n")
                line = []
        if line:
            file.write("".join(line) + "\n")

    with open(output_path, "w") as f:
        f.write("*NODE\n")
        for row in node_section:
            node_id = int(row[0])
            x, y, z = row[1:4]
            tc = int(row[4])
            rc = int(row[5])
            #node_id width=8, then one space, then space-delimited x, y, z
            f.write(f"{node_id:8d} {x:.9E} {y:.9E} {z:.9E}{tc:8d}{rc:8d}\n")

        f.write("*ELEMENT_SOLID\n")
        for row in element_section:
            #10 integer fields with width=8
            f.write("".join(f"{val:8d}" for val in row) + "\n")

        #Sensor element set
        if sensor_elements:
            f.write("*SET_SOLID_LIST\n")
            f.write(f"{int(sensor_set_id)}\n")
            write_set_section(f, sensor_elements)

        #Tracer element set
        if tracer_elements:
            f.write("*SET_SOLID_LIST\n")
            f.write(f"{int(tracer_elset_id)}\n")
            write_set_section(f, tracer_elements)

        #Tracer node set
        if tracer_nodes:
            f.write("*SET_NODE_LIST\n")
            f.write(f"{int(tracer_nset_id)}\n")
            write_set_section(f, tracer_nodes)

        f.write("*END\n")


# -----------------------------
# Main
# -----------------------------
def main(element_size, outer_dims):
    nodes = generate_nodes(element_size, outer_dims)
    if nodes is None:
        return

    node_section = add_constraints(nodes, FIXED_COORDS)
    element_section = generate_elements(node_section, element_size, outer_dims, EXPL_RADIUS_CM)

    sensor_elements = define_sensor_elements(
        node_section, element_section, element_size, outer_dims, SENSOR_OFFSET_EL
    )
    tracer_elements = define_tracer_elements(node_section, element_section)
    tracer_nodes = define_tracer_nodes(element_section, tracer_elements)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILENAME)

    format_sections_into_file(
        node_section, element_section,
        sensor_elements, SENSOR_SET_ID,
        tracer_elements, TRACER_ELSET_ID,
        tracer_nodes, TRACER_NSET_ID,
        output_path
    )


# -----------------------------
# Script
# -----------------------------
if __name__ == '__main__':
    element_size = float(ELEMENT_SIZE_CM)
    outer_dims = (float(REGION_X_DIM), float(REGION_Y_DIM), float(REGION_Z_DIM))

    if not check_integer_num_elements(element_size, outer_dims):
        sys.exit(1)

    main(element_size, outer_dims)