# Script to parse LS-DYNA mesh files and save nodes and elements to separate CSV files
# Haena Lee, June 2025

import re
import pandas as pd

#HELPER FUNCTIONS
#Get section starting from keyword (e.g., *NODE) up to next section or blank lines
def extract(content, keyword):
    lines = content.splitlines()
    section = []
    reading = False
    for line in lines:
        if line.strip().startswith("*"):    #if it's a keyword
            reading = line.strip().upper().startswith(keyword)
            continue
        if reading and line.strip():        #if we're reading and the line isn't blank
            section.append(line)
        elif reading and not line.strip():  #if it's a blank line
            break
    return section

#Parse *NODE section into dataframe with columns: node ID, x, y, z, tc, rc
def parse_nodes(section):
    #Regex for each column entry
    pattern = re.compile(
        r'^\s*(\d+)\s+'          #node ID
        r'([-+0-9.Ee]+)\s+'      #x coord
        r'([-+0-9.Ee]+)\s+'      #y coord
        r'([-+0-9.Ee]+)\s+'      #z coord
        r'(\d+)\s+'              #tc
        r'(\d+)'                 #rc
    )

    #Save the lines that match the regex pattern as lists of tuples and convert to dataframe
    matches = [pattern.match(line) for line in section if pattern.match(line)]
    data = [m.groups() for m in matches]
    df = pd.DataFrame(data, columns=['Node ID', 'X', 'Y', 'Z', 'TC', 'RC'])

    #Convert columns to correct types
    df = df.astype({
        'Node ID': int,
        'X': float,
        'Y': float,
        'Z': float,
        'TC': int,
        'RC': int
    })
    return df

#Parse *ELEMENT_SOLID section into dataframe with columns: element ID, part ID, 8 columns of node IDs
def parse_elements(section):
    #If there are 10 integers in the line, save the lists and convert to dataframe
    matches = [re.findall(r'\d+', line) for line in section if len(re.findall(r'\d+',line))==10]
    df = pd.DataFrame(matches, columns=['Element ID', 'Part ID', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'])
    
    #Convert all columns (from strings) to integers
    df = df.astype(int)    
    return df


#MAIN
#Parse mesh file and save nodes to nodes.csv and elements to elements.csv
def main(file_path):
    #Read file
    with open(file_path, 'r') as f:
        content = f.read()

    #Extract, parse, save *NODE section to nodes.csv
    node_section = extract(content, '*NODE')
    node_df = parse_nodes(node_section)
    node_df.to_csv(folder_path+'nodes.csv', index=False)

    #Extract, parse, save *ELEMENT_SOLID section to elements.csv
    elmt_section = extract(content, '*ELEMENT_SOLID')
    elmt_df = parse_elements(elmt_section)
    elmt_df.to_csv(folder_path+'elements.csv', index=False)


#SCRIPT
if __name__ == '__main__':
    #SPECIFY: folder and mesh file paths
    folder_path = '/Users/haenalee/Documents/Research/lsdynaEx/'
    main(folder_path+'fine.inc')
