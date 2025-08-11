# Script to generate the mesh file for a mesh with a cutout region; see https://lsdyna.ansys.com/underwater-f/ for an example
# Assumptions: mesh's bottom left corner is at the origin, thin thickness of z=1, bottom edge of cutout region is on the x axis,
# inputs except element_size are integers
# Haena Lee, July 2025

import numpy as np

#HELPER FUNCTIONS
#Generate the node IDs and coordinates; ordered in the same manner as in 'fine.inc'
def generate_nodes(element_size, outer_dims, expl_dims, cutout_dims, cutout_offset):
    xf, yf, zf = outer_dims
    xf_expl, yf_expl, zf_expl = expl_dims
    xi_cutout_offset = cutout_offset[0]
    yf_cutout = cutout_dims[1]
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
    nxi_c_o = to_index(xi_cutout_offset)
    nyf_c = to_index(yf_cutout)

    def add_node(xi,yi,zi):
        #append coordinates scaled by element_size
        nodes.append((xi*element_size, yi*element_size, zi*element_size))

    #Explosive region, including boundaries
    for z in range(nzf_e+1):
        for y in range(nyf_e+1):
            for x in range(nxf_e+1):
                add_node(x,y,z)

    #Region above explosive region to cutout height
    for z in range(nzf+1):
        for y in range(nyf_e+1, nyf_c+1):
            for x in range(0, nxf_e+1):
                add_node(x,y,z)

    #Region further above
    for z in range(nzf+1):
        for y in range(nyf_c+1, nyf+1):
            for x in range(0, nxf_e+1):
                add_node(x,y,z)

    #Region to the right of explosive region
    for z in range(nzf+1):
        for y in range(0, nyf_e+1):
            for x in range(nxf_e+1, nxi_c_o+1):
                add_node(x,y,z)

    #Region above to cutout height
    for z in range(nzf+1):
        for y in range(nyf_e+1, nyf_c+1):
            for x in range(nxf_e+1, nxi_c_o+1):
                add_node(x,y,z)

    #Region further above
    for z in range(nzf+1):
        for y in range(nyf_c+1, nyf+1):
            for x in range(nxf_e+1, nxi_c_o+1):
                add_node(x,y,z)

    #Region above the cutout
    for z in range(nzf+1):
        #where y = nyf_c
        for x in range(nxi_c_o+1, nxf+1):
            add_node(x, nyf_c, z)
        #then above it
        for y in range(nyf_c+1, nyf+1):
            for x in range(nxi_c_o+1, nxf+1):
                add_node(x, y, z)

    nodes = np.array(nodes, dtype=float)
    node_IDs = np.arange(1, nodes.shape[0]+1).reshape(-1, 1)    #generate column of node IDs
    nodes = np.hstack((node_IDs, nodes))
    return nodes


#Add translational and rotational constraints to nodes
def add_constraints(nodes, outer_dims, cutout_dims, cutout_offset, fixed_coords):
    xf, yf, _ = outer_dims
    yf_cutout = cutout_dims[1]
    xi_cutout_offset = cutout_offset[0]
    coords = nodes[:,1:4]
    tc = np.full((coords.shape[0],), 3)  #fill tc column with tc=3 (constrained in z disp) - will be updated
    rc = np.full((coords.shape[0],), 7)  #fill rc column with rc=7 (constrained in xyz rot) - won't touch

    for i, (x,y,z) in enumerate(coords):
        #if node is on outer edges parallel to x axis but that are not the top edges
        if np.isclose(y,0) or np.isclose(y,yf_cutout) and x > xi_cutout_offset and not np.isclose(y,yf):
            tc[i]=5     #constrain y and z disp

        #if node is on outer edges parallel to y axis but that are not the rightmost edges
        if np.isclose(x,0) or np.isclose(x,xi_cutout_offset) and y < yf_cutout and not np.isclose(x,xf):
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
def generate_elements(nodes, outer_dims, expl_dims, cutout_dims, cutout_offset):
    xf, yf, zf = map(int, outer_dims)
    xf_expl, yf_expl, zf_expl = map(int, expl_dims)
    yf_cutout = int(cutout_dims[1])
    xi_cutout_offset = int(cutout_offset[0])
    part_nonexpl = 1    #part ID of the non-explosive region
    part_expl = 2
    elements = []

    #Dictionary to map coordinates to node IDs
    IDs = nodes[:,0].astype(int)
    coords = np.round(nodes[:,1:4].astype(float), 9)   #round all coordinates to 9 decimal places
    coords_to_id = {}
    for nodeID, xyz in zip(IDs, coords):
        x, y, z = xyz
        key = (x, y, z)
        coords_to_id[key] = int(nodeID)


    #Define the 8 vertices of each element and return their corresponding node IDs
    def element_node_IDs(i,j,k):
        vertices = [
            (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k),
            (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1),
        ]
        try:
            return [coords_to_id[v] for v in vertices]   #try finding the node ID for each vertex
        except KeyError:
            return None

    #Check if element is in the explosive region
    def is_expl(x,y,z):
        return (0 <= x < xf_expl) and (0 <= y < yf_expl) and (0 <= z < zf_expl)

    #Explosive region, including boundaries
    for z in range(zf_expl):
        for y in range(yf_expl):
            for x in range(xf_expl):
                ns = element_node_IDs(x,y,z)    #get the node IDs at the 8 vertices
                if ns is None:      #if any of the nodes at the vertices don't exist
                    continue
                elements.append([part_expl, *ns])

    #Region above explosive region
    for y in range(yf_expl, yf):
        for x in range(0, xf_expl):
            for z in range(0, zf_expl):
                ns = element_node_IDs(x,y,z)
                if ns is None: 
                    continue
                part = part_expl if is_expl(x,y,z) else part_nonexpl
                elements.append([part, *ns])

    #Region to the right of explosive region
    for y in range(0, yf):
        x_start = xf_expl
        x_end = xf-1
        if y >= yf_cutout:
            x_end = max(xf_expl-1, xi_cutout_offset-1)  # stop before cutout columns
        for x in range(x_start, x_end+1):
            for z in range(0, zf):
                ns = element_node_IDs(x,y,z)
                if ns is None: 
                    continue
                part = part_expl if is_expl(x,y,z) else part_nonexpl
                elements.append([part, *ns])

    #Region above cutout
    if yf_cutout < yf and xi_cutout_offset < xf:
        for y in range(yf_cutout, yf):
            for x in range(xi_cutout_offset, xf):
                for z in range(0, zf):
                    ns = element_node_IDs(x,y,z)
                    if ns is None:
                        continue
                    part = part_expl if is_expl(x,y,z) else part_nonexpl
                    elements.append([part, *ns])

    elements = np.array(elements, dtype=int)
    element_IDs = np.arange(1, elements.shape[0]+1).reshape(-1, 1)  #generate column of element IDs
    elements = np.hstack((element_IDs, elements))
    return elements


#Format the node and element sections into the output file, in the same manner as 'fine.inc'
def format_sections_into_file(node_section, element_section, output_filename):
    with open(output_filename, "w") as f:
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
        
        f.write("*END\n")


#MAIN
def main(output_filename, element_size, outer_dims, cutout_dims, cutout_offset, expl_dims, fixed_coords):
    nodes = generate_nodes(element_size, outer_dims, expl_dims, cutout_dims, cutout_offset)
    node_section = add_constraints(nodes, outer_dims, cutout_dims, cutout_offset, fixed_coords)
    element_section = generate_elements(node_section, outer_dims, expl_dims, cutout_dims, cutout_offset)

    format_sections_into_file(node_section, element_section, output_filename)
    print(f"Mesh written to {output_filename}.")


#SCRIPT
if __name__ == '__main__':
    output_filename = input("Enter output file name (e.g., output.inc): ").strip()
    element_size = float(input("Enter element size (e.g., 1): "))
    
    def get_tuple(prompt):
        return tuple(map(float, input(prompt).strip().split()))

    outer_dims = get_tuple("Enter outer region dimensions (e.g., 100 100 1): ")
    cutout_dims = get_tuple("Enter cutout region dimensions (e.g., 60 50 1): ")
    cutout_offset = get_tuple("Enter lower corner cutout offset from origin (e.g., 40 0 0): ")
    expl_dims = get_tuple("Enter explosive region dimensions (e.g., 8 8 1): ")

    fixed_coords = []
    fixed_coords_input = input("Enter fixed node coordinates, separated by commas (e.g., 0 0 0, 0 0 1, 40 0 0, 40 0 1, 40 50 0, 40 50 1): ").strip()
    if fixed_coords_input:
        for group in fixed_coords_input.split(','):
            coord = tuple(map(float, group.strip().split()))
            fixed_coords.append(coord)

    main(output_filename, element_size, outer_dims, cutout_dims, cutout_offset, expl_dims, fixed_coords)