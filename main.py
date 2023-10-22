import contextlib
import math
import random

import numpy as np
import plotly.graph_objects as go

from scipy.spatial import cKDTree as KDTree
from sklearn.cluster import KMeans


def generate_cluster(distance, center):
    x = np.random.uniform(center[0], center[0] + distance, size=(100,))
    y = np.random.normal(center[1], distance, size=(100,))
    return (x, y)


def create_spiral(max_radius, arm_count, beta):
    radius_pct = random.betavariate(10, 10)
    radius = radius_pct * max_radius

    base_angle = math.log(radius) / (beta)

    angle_pct = random.betavariate(4, 4)
    arm_num = random.randint(0, arm_count - 1)
    arm_angle = (angle_pct + arm_num) * 2 * math.pi / arm_count

    angle = base_angle + arm_angle
    x = math.cos(angle) * radius
    y = math.sin(angle) * radius

    return (x, y)


def create_inner(max_radius):
    radius_pct = random.betavariate(1.5, 4)
    radius = radius_pct * max_radius/2

    angle = random.uniform(0, 2 * math.pi)

    x = math.cos(angle) * radius
    y = math.sin(angle) * radius

    return (x, y)


def create_outer(max_radius):
    radius_pct = random.betavariate(2, 2)
    radius = radius_pct * max_radius*0.9

    angle = random.uniform(0, 2 * math.pi)

    x = math.cos(angle) * radius
    y = math.sin(angle) * radius

    return (x, y)

def main():
    np.random.seed(10)
    random.seed(10)
    points = {}

    # For large sizes zooming out gives better view of the shape
    SIZE = 1000
    SYSTEMS = 6000
    MIN_DISTANCE = 6
    ARM_COUNT = 10
    num_regions = ARM_COUNT + 1
    max_connections = 4

    # Connection code is laggy for larger maps.
    connections_enabled = False

    # Color for the regions
    colors = ['blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 'magenta', 'lime', 'pink', 'brown', "lavender", "mint"]
    i = 0
    for _ in range(int(SYSTEMS*0.7)):

        s = create_spiral(SIZE, ARM_COUNT, 4)
        points[i] = {"coordinates": s, "type": "spiral", "connections": []}
        i += 1

    for _ in range(int(SYSTEMS*0.2)):
        s = create_inner(SIZE)
        points[i] = {"coordinates": s, "type": "inner", "connections": []}
        i += 1


    for _ in range(int(SYSTEMS*0.1)):
        s=create_outer(SIZE)
        points[i] = {"coordinates": s, "type": "filler", "connections": []}
        i += 1

    coords = np.asarray([value["coordinates"] for value in points.values()])
    # Clean the map
    tree = KDTree(coords)
    distances, index_list = tree.query(coords, k=8, eps=0.1)


    # Create the clusters
    kmeans = KMeans(n_clusters=num_regions)

    kmeans.fit(coords)
    cluster_assignments = kmeans.predict(coords)

    # Save the cluster to dict
    for index in points:
        points[index]["cluster"] = cluster_assignments[index]


    for i, query_point in enumerate(coords):
        nearest_neighbors = index_list[i]
        nearest_distances = distances[i]

        combined = list(zip(nearest_distances, nearest_neighbors))
        itself = combined.pop(0)[1]
        if itself not in points:
            continue

        choices = [1, 2, 3, 4, 5]
        weights = [0.10, 0.35, 0.35, 0.15, 0.1]

        num = max(0, random.choices(choices, weights)[0]-len(points.get(itself, {}).get("connections", [])))
        closest = None
        closest_distance = math.inf
        not_chosen = []
        for d, n in combined:
            if d < MIN_DISTANCE:
                with contextlib.suppress(Exception):
                    points.pop(n)

            # If the system has been removed do nothing.
            if n not in points or itself not in points:
                continue

            # Skip systems with more than 4 connections
            if len(points[n]["connections"]) >= max_connections:
                continue

            # If no connections find the closest legal point to connect
            if not points[itself]["connections"] and d < closest_distance:
                closest_distance = d
                if closest is not None:
                    not_chosen.append((n, (random.randint(9,11)/10)/(d) * 1.5 if points[n]["cluster"] == points[itself]["cluster"] else 1))
                closest = n
            else:
                not_chosen.append((n, (random.randint(9,11)/10)/(d) * 1.5 if points[n]["cluster"] == points[itself]["cluster"] else 1))

        # If closest was chosen then remove one possible system
        if closest is not None:
            num = max(0, num - 1)

        if not points[itself]["connections"]:
            points[itself]["connections"].append(n)

        points[itself]["connections"].append(closest)
        sum_of_weights = sum([x[1] for x in not_chosen])
        normalized_weights = [(x[1] / sum_of_weights) for x in not_chosen]
        systems = [(x[0]) for x in not_chosen]

        if systems:
            connections = random.choices(systems, weights=normalized_weights, k=num)
            for x in connections:
                if not len(points[x]["connections"]) >= max_connections:
                    points[itself]["connections"].append(x)



    connections_trace = []
    locations_trace = []

    for key, value in points.items():
        x, y = value["coordinates"]
        text = f"ID: {key}<br>Type: {value['type']}<br>Cluster: {value['cluster']}"
        locations_trace.append(go.Scatter(
            x=[x],
            y=[y],
            text=text,
            mode="markers",
            marker=dict(size=10, color=colors[int(value["cluster"])]),
            textfont=dict(size=10, color='black'),
            name=f"ID: {key}"
        ))

        if connections_enabled:
            for i, connection_id in enumerate(value["connections"]):
                connection_data = points.get(connection_id)
                if connection_data:
                    connection_x, connection_y = connection_data["coordinates"]

                    connections_trace.append(go.Scatter(
                        x=[x, connection_x],
                        y=[y, connection_y],
                        mode='lines',
                        line=dict(width=2, color=f'rgba(255,0,0,255)'),
                        name=f"Connection: {key} to {connection_id}",
                    ))

    # Combine the traces and layout to create the figure
    fig = go.Figure(data=connections_trace + locations_trace)

    # Show the interactive map
    fig.show()


if __name__ == '__main__':
    main()
