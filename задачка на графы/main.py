import random
import math
from collections import deque
from typing import List, Tuple, Optional


def generate_cities(n: int, min_coord: int = 0, max_coord: int = 100) -> List[Tuple[int, int]]:
    return [(random.randint(min_coord, max_coord), random.randint(min_coord, max_coord)) for _ in range(n)]


def calculate_distance(city1: Tuple[int, int], city2: Tuple[int, int]) -> float:
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])


def bfs_tsp(cities):
    n = len(cities)
    if n == 0:
        return 0, []
    if n == 1:
        return 0, [0, 0]
    if n == 2:
        d = ((cities[0][0] - cities[1][0])**2 + (cities[0][1] - cities[1][1])**2)**0.5
        return d * 2, [0, 1, 0]

    best = math.inf

    best_route = []

    from collections import deque
    q = deque()
    q.append([0, [0], 0])

    while len(q) > 0:
        cur, visited, dist = q.popleft()

        if len(visited) == n:
            back = ((cities[cur][0] - cities[0][0])**2 + (cities[cur][1] - cities[0][1])**2)**0.5
            total = dist + back
            if total < best:
                best = total
                best_route = visited + [0]
            continue

        for i in range(n):
            if i in visited:
                continue

            d_to_next = ((cities[cur][0] - cities[i][0])**2 + (cities[cur][1] - cities[i][1])**2)**0.5
            new_dist = dist + d_to_next
            new_visited = visited + [i]

            q.append([i, new_visited, new_dist])

    if not best_route:
        best_route = [0, 0]
    return best, best_route


def main():
    random.seed()
    n = 8
    cities = generate_cities(n)

    print("Координаты городов:")
    for i, (x, y) in enumerate(cities):
        print(f"Город {i}: ({x}, {y})")

    distance, path = bfs_tsp(cities)
    print(f"\nОптимальное расстояние: {distance:.2f}")
    print("Оптимальный путь:", " → ".join(map(str, path)))


if __name__ == "__main__":
    main()