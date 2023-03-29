from json import dumps


def dijkstra(graph: list, start_vertex: int) -> list:
    num_vertices = len(graph)
    way = {i: "_" for i in range(num_vertices)}
    way[start_vertex] = start_vertex

    distances = [float('inf')] * num_vertices
    distances[start_vertex] = 0

    unvisited = set(range(num_vertices))

    # print(unvisited)
    # print(dumps(way, indent=4))
    # print(distances)

    while unvisited:
        current_vertex = min(unvisited, key=lambda vertex: distances[vertex])

        for neighbor in range(num_vertices):
            if graph[current_vertex][neighbor] == 0:
                continue

            weight = graph[current_vertex][neighbor]
            distance = distances[current_vertex] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                way[neighbor] = current_vertex

        unvisited.remove(current_vertex)

        # print(f'vertex: {current_vertex}')
        # print(unvisited)
        # print(dumps(way, indent=4))
        # print(distances)

    return way, distances


def print_way(way, vertex):
    if way[vertex] != vertex:
        print_way(way, way[vertex])
        print(" -> ", end='')

    print(vertex + 1, end="")


def main() -> None:
    graph = [
        # 1  2  3  4  5  6
        [0, 0, 12, 11, 0, 3],  # 1
        [0, 0, 0, 0, 5, 5],  # 2
        [12, 0, 0, 0, 0, 1],  # 3
        [11, 0, 0, 0, 6, 0],  # 4
        [0, 5, 10, 6, 0, 0],  # 5
        [3, 5, 1, 0, 0, 0]  # 6
    ]

    way, distances = dijkstra(
        graph=graph,
        start_vertex=0,
    )

    vertex = 5
    print(f"The way to the top {vertex + 1}: ", end="")
    print_way(way, vertex)
    print()
    print(f"distance: {distances[vertex]}")


if __name__ == '__main__':
    main()
