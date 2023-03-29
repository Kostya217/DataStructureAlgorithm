def change_color(to_visit, new_color, old_color):
    for i in range(len(to_visit)):
        if to_visit[i] == old_color:
            to_visit[i] = new_color


def search_min(graf, visited, to_visit) -> list:
    min_num = max(max(graf))
    x, y = 0, 0
    for i in range(len(graf)):
        for j in range(i + 1, len(graf[i])):
            if (graf[i][j] > 0) and (graf[i][j] < min_num) and ((i, j) not in visited) and (to_visit[i] != to_visit[j]):
                min_num = graf[i][j]
                x, y = i, j

    change_color(
        to_visit=to_visit,
        new_color=x + 1,
        old_color=to_visit[y],
    )
    return [min_num, (x, y)]


def prim(graf: list) -> tuple:
    to_visit = [i for i in range(1, len(graf) + 1)]
    visited = []
    result = []
    for _ in range(1, len(graf)):
        weight, top = search_min(graf, visited, to_visit)
        result.append(weight)
        visited.append(top)
    return visited, result


def main() -> None:
    graf = [
        # 1  2  3  4  5  6
        [0, 0, 4, 11, 0, 3],  # 1
        [0, 0, 0, 0, 5, 5],  # 2
        [4, 0, 0, 0, 0, 1],  # 3
        [11, 0, 0, 0, 6, 0],  # 4
        [0, 5, 10, 6, 0, 0],  # 5
        [3, 0, 1, 0, 0, 0]  # 6
    ]

    tops, result = prim(graf)
    tops = [((i + 1), (j + 1)) for i, j in tops]
    print(f"minimal frame tree: {tops}")
    print(f"edges: {result}")
    print(f"sum of edges: {sum(result)}")


if __name__ == '__main__':
    main()
