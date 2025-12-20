import json

# Define the hairstyle-to-face-shape mapping
data = {
    "Undercut": ["Square", "Round", "Triangle"],
    "Pompadour": ["Diamond", "Triangle", "Oblong"],
    "Crewcut": ["Square", "Rectangular", "Round"],
    "Slicked Back": ["Diamond", "Triangle", "Oblong"],
    "Fringe": ["Round", "Triangle", "Diamond"],
    "Side Part": ["Square", "Rectangular", "Round"],
    "Buzzcut": ["Square", "Diamond", "Rectangular"]
}

# Write the data to a JSON file
with open("map.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("map.json has been created successfully.")
