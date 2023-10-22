data = {
    1: {"sys": "value1", "coordinates": (1.0, 2.0)},
    2: {"sys": "value2", "coordinates": (3.0, 4.0)},
    3: {"sys": "value3", "coordinates": (5.0, 6.0)}
}

coordinates = [value["coordinates"] for value in data.values()]
numeric_coordinates = [[float(coord) for coord in coords] for coords in coordinates]
indices = list(data.keys())
print(coordinates, indices)