import xml.etree.ElementTree as ET
import copy

input_file = '/home/matteo/ethrc/yam/integration/piper-robot-server/URDF/yam/dual_yam.urdf'
output_file = '/home/matteo/ethrc/yam/integration/piper-robot-server/URDF/yam/dual_yam_generated.urdf'

tree = ET.parse(input_file)
root = tree.getroot()

# Find all links and joints
links = root.findall('link')
joints = root.findall('joint')

new_links = []
new_joints = []

# Clone links
for link in links:
    new_link = copy.deepcopy(link)
    old_name = new_link.get('name')
    new_name = f"arm2_{old_name}"
    new_link.set('name', new_name)
    new_links.append(new_link)

# Clone joints
for joint in joints:
    new_joint = copy.deepcopy(joint)
    old_name = new_joint.get('name')
    new_name = f"arm2_{old_name}"
    new_joint.set('name', new_name)
    
    parent = new_joint.find('parent')
    if parent is not None:
        parent.set('link', f"arm2_{parent.get('link')}")
        
    child = new_joint.find('child')
    if child is not None:
        child.set('link', f"arm2_{child.get('link')}")
        
    new_joints.append(new_joint)

# Create the connecting joint
connecting_joint = ET.Element('joint')
connecting_joint.set('name', 'base_to_arm2_base_joint')
connecting_joint.set('type', 'fixed')

origin = ET.Element('origin')
origin.set('xyz', '0 0.5 0')
origin.set('rpy', '0 0 0')
connecting_joint.append(origin)

parent = ET.Element('parent')
parent.set('link', 'base_link')
connecting_joint.append(parent)

child = ET.Element('child')
child.set('link', 'arm2_base_link')
connecting_joint.append(child)

# Append new elements to root
# We want to keep the original elements and add the new ones
# But we need to be careful about order if it matters (usually doesn't for URDF parsers, but nice to group)

root.append(ET.Comment(' Second Arm '))
root.append(connecting_joint)

for link in new_links:
    root.append(link)

for joint in new_joints:
    root.append(joint)

# Write to file
tree.write(output_file, encoding='utf-8', xml_declaration=True)

# Read back and print to verify (or just read it in next step)
with open(output_file, 'r') as f:
    print(f.read())
