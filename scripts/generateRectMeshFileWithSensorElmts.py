# Script to generate the mesh file for a rectangular mesh with sensor elements for automatic simulation termination
# Assumptions: mesh's bottom left corner is at the origin, thin thickness of z=1, inputs except element_size are integers, 
# part ID of non-explosive is 1 and ID of explosive is 2, region is rectangular
# Haena Lee, August 2025

import numpy as np
import os

#HELPER FUNCTIONS
#Generate the node IDs and coordinates; ordered in the same manner as in 'fine.inc'
def generate_nodes(element_size, outer_dims, expl_dims):
    xf, yf, zf = outer_dims
    xf_expl, yf_expl, zf_expl = expl_dims
    tol = 1e-10
    nodes = []

    #Check that there are an integer # of elements in the explosive region
    nx_expl = xf_expl/element_size
    ny_expl = yf_expl/element_size

    def is_int(x):  #check if x is an integer within tolerance
        return abs(x-round(x)) <= tol

    if not (is_int(nx_expl) and is_int(ny_expl)):   #if there's not an integer # of elements
        print(
            f"There are {nx_expl:.2f} elements in the x dimension and "
            f"{ny_expl:.2f} elements in the y dimension of the explosive region.\n"
            "Please choose a different element size or explosive dimensions."
        )
        return None     #quit early

    #Convert all lengths to indices, rounding to nearest int
    def to_index(L):
        return int(round(L/element_size))
    nxf = to_index(xf)
    nyf = to_index(yf)
    nzf = to_index(zf)
    nxf_e = to_index(xf_expl)
    nyf_e = to_index(yf_expl)
    nzf_e = to_index(zf_expl)

    def add_node(xi,yi,zi):
        #append coordinates scaled by element_size
        nodes.append((xi*element_size, yi*element_size, zi*element_size))

    #Explosive region, including boundaries
    for z in range(nzf_e+1):
        for y in range(nyf_e+1):
            for x in range(nxf_e+1):
                add_node(x,y,z)

    #Region above explosive region
    for z in range(nzf+1):
        for y in range(nyf_e+1, nyf+1):
            for x in range(0, nxf_e+1):
                add_node(x,y,z)

    #Region to the right of explosive region
    for z in range(nzf+1):
        for y in range(0, nyf_e+1):
            for x in range(nxf_e+1, nxf+1):
                add_node(x,y,z)

    #Region above
    for z in range(nzf+1):
        for y in range(nyf_e+1, nyf+1):
            for x in range(nxf_e+1, nxf+1):
                add_node(x,y,z)

    nodes = np.array(nodes, dtype=float)
    node_IDs = np.arange(1, nodes.shape[0]+1).reshape(-1, 1)    #generate column of node IDs
    nodes = np.hstack((node_IDs, nodes))
    return nodes


#Add translational and rotational constraints to nodes
def add_constraints(nodes, outer_dims, fixed_coords):
    xf, yf, _ = outer_dims
    coords = nodes[:,1:4]
    tc = np.full((coords.shape[0],), 3)  #fill tc column with tc=3 (constrained in z disp) - will be updated
    rc = np.full((coords.shape[0],), 7)  #fill rc column with rc=7 (constrained in xyz rot) - won't touch

    for i, (x,y,z) in enumerate(coords):
        #if node is on outer edges parallel to x axis but that are not the top edges
        if np.isclose(y,0) or np.isclose(y,yf) and x > xf and not np.isclose(y,yf):
            tc[i]=5     #constrain y and z disp

        #if node is on outer edges parallel to y axis but that are not the rightmost edges
        if np.isclose(x,0) or np.isclose(x,xf) and y < yf and not np.isclose(x,xf):
            tc[i]=6     #constrain z and x disp

        #if node is one of the specified fixed nodes
        for fx, fy, fz in fixed_coords:
            if np.isclose([x,y,z], [fx,fy,fz]).all():
                tc[i]=7     #constrain xyz disp
                break

    #Add tc and rc columns
    constraints = np.column_stack((tc, rc))
    nodes = np.hstack((nodes, constraints))
    return nodes


#Generate the hexahedral elements; ordered in the same manner as in 'fine.inc'
def generate_elements(nodes, element_size, outer_dims, expl_dims):
    xf, yf, zf = map(int, outer_dims)
    xf_expl, yf_expl, zf_expl = map(int, expl_dims)
    part_nonexpl = 1    #part ID of the non-explosive region
    part_expl = 2
    elements = []

    #Create node ID map with columns: node ID, x, y, z
    nodes = np.asarray(nodes)
    ids = nodes[:,0].astype(int)
    xs = nodes[:,1].astype(float)
    ys = nodes[:,2].astype(float)
    zs = nodes[:,3].astype(float)

    #Convert physical coordinates to integer grid indices (i,j,k) based on element size.
    scale = 1.0/float(element_size)     #multiply by this instead of dividing to prevent floating pt errors
    tol = 1e-10
    ix = np.rint(xs*scale+tol).astype(int)
    iy = np.rint(ys*scale+tol).astype(int)
    iz = np.rint(zs*scale+tol).astype(int)
    indices = np.stack([ix, iy, iz], axis=1)
    
    #Dictionary to get node ID from integer indices
    index_to_id = {}
    for idx, nid in zip(indices, ids):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        index_to_id[key] = int(nid)

    def nid(i,j,k):
        return index_to_id[(i,j,k)]   #corresponding node ID for the integer indices

    #Convert lengths to grid indices, rounding to nearest int
    #E.g., if xf = 100 and element_size = 0.5, to_index(100) -> 200.
    def to_index(L):
        return int(round(L/element_size))
    nxf = to_index(xf)
    nyf = to_index(yf)
    nzf = to_index(zf)
    nxf_e = to_index(xf_expl)
    nyf_e = to_index(yf_expl)
    nzf_e = to_index(zf_expl)

    #Define the 8 vertices of each element and return their corresponding node IDs
    def element_node_IDs(i,j,k):
        try:        #try finding the node ID for each vertex
            return [
                nid(i,j,k), nid(i+1,j,k), nid(i+1,j+1,k), nid(i,j+1,k),
                nid(i,j,k+1), nid(i+1,j,k+1), nid(i+1,j+1,k+1), nid(i,j+1,k+1)
            ]
        except KeyError:
            return None

    #Explosive region, including boundaries
    for z in range(nzf_e):
        for y in range(nyf_e):
            for x in range(nxf_e):
                ns = element_node_IDs(x,y,z)    #get the node IDs at the 8 vertices
                if ns is None:      #if any of the nodes at the vertices don't exist
                    continue
                elements.append([part_expl, *ns])

    #Region above explosive region
    for z in range(nzf):
        for y in range(nyf_e, nyf):
            for x in range(0, nxf_e):
                ns = element_node_IDs(x,y,z)
                if ns is None: 
                    continue
                elements.append([part_nonexpl, *ns])

    #Region to the right of explosive region
    for z in range(nzf):
        for y in range(0, nyf):
            for x in range(nxf_e, nxf):
                ns = element_node_IDs(x,y,z)
                if ns: elements.append([part_nonexpl, *ns])

    elements = np.asarray(elements, dtype=int)
    element_IDs = np.arange(1, elements.shape[0]+1, dtype=int).reshape(-1, 1)  #generate column of element IDs
    elements = np.hstack((element_IDs, elements))
    return elements


#Return the sensor elements offset from the top and right boundaries by the # of elements specified by sensor_offset; assuming region's thickness is z=1
def define_sensor_elements(node_section, element_section, element_size, outer_dims, sensor_offset):
    xf, yf, _ = map(float, outer_dims)
    es = float(element_size)
    nxf = int(round(xf/es))
    nyf = int(round(yf/es))

    #X and y coordinates of the top and right edges of the sensor elements
    sensor_y_coord = (nyf-sensor_offset) * es
    sensor_x_coord = (nxf-sensor_offset) * es

    #Dictionary to get coordinates from node IDs
    nodeID_to_coords = {}
    for row in node_section:
        node_id = int(row[0])
        nodeID_to_coords[node_id] = tuple(map(float, row[1:4]))

    sensor_elements = set()      #use set to avoid duplicate elements (e.g., the one where the row and column intersect)
    tol = 1e-10

    for row in element_section:
        eid = int(row[0])
        nids = [int(v) for v in row[2:10]]
        x_coords = [nodeID_to_coords[n][0] for n in nids]
        y_coords = [nodeID_to_coords[n][1] for n in nids]

        on_sensor_row = abs(max(y_coords) - sensor_y_coord) <= tol
        on_sensor_col = abs(max(x_coords) - sensor_x_coord) <= tol

        if on_sensor_row or on_sensor_col:
            sensor_elements.add(eid)

    return sensor_elements


#Format the node and element sections into the output file, in the same manner as 'fine.inc'
def format_sections_into_file(node_section, element_section, sensor_elements, sensor_set_id, output_path):
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
        
        #Sensor element set (union of top + right or top + inner-right for L-shape)
        if sensor_elements:
            f.write("*SET_SOLID_LIST\n")
            f.write(f"{int(sensor_set_id)}\n")
            line = []
            for i, eid in enumerate(sensor_elements, 1):
                line.append(f"{int(eid):10d}")
                if i % 10 == 0:
                    f.write("".join(line) + "\n")
                    line = []
            if line:
                f.write("".join(line) + "\n")

        f.write("*END\n")


#MAIN
def main(output_filename, element_size, outer_dims, expl_dims, fixed_coords, sensor_offset, sensor_set_id):
    nodes = generate_nodes(element_size, outer_dims, expl_dims)
    node_section = add_constraints(nodes, outer_dims, fixed_coords)
    element_section = generate_elements(node_section, element_size, outer_dims, expl_dims)
    sensor_elements = define_sensor_elements(node_section, element_section, element_size, outer_dims, sensor_offset)

    script_dir = os.path.dirname(os.path.abspath(__file__))     #directory of this script
    output_path = os.path.join(script_dir, output_filename)
    format_sections_into_file(node_section, element_section, sensor_elements, sensor_set_id, output_path)
    print(f"Mesh written to {output_filename}.")


#SCRIPT
if __name__ == '__main__':
    output_filename = input("Enter output file name (e.g., output.inc): ").strip()
    element_size = float(input("Enter element size (e.g., 1): "))
    
    def get_tuple(prompt):
        return tuple(map(float, input(prompt).strip().split()))

    outer_dims = get_tuple("Enter outer region dimensions (e.g., 100 100 1): ")
    expl_dims = get_tuple("Enter explosive region dimensions (e.g., 8 8 1): ")

    fixed_coords = []
    fixed_coords_input = input("Enter fixed node coordinates, separated by commas (e.g., 0 0 0, 0 0 1): ").strip()
    if fixed_coords_input:
        for group in fixed_coords_input.split(','):
            coord = tuple(map(float, group.strip().split()))
            fixed_coords.append(coord)

    sensor_offset = float(input("Enter the number of elements to offset the sensor elements by from the top and right boundaries (e.g., 1): "))
    sensor_set_id = float(input("Enter sensor element set ID (e.g., 9001): "))


    main(output_filename, element_size, outer_dims, expl_dims, fixed_coords, sensor_offset, sensor_set_id)
