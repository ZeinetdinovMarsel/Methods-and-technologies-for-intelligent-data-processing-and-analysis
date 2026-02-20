import numpy as np
import random as rand
import time

n = 8
chess_table = np.zeros((n, n), dtype=int)


class QueenAgent:
    def __init__(self, row):
        self.row = row
        self.col = rand.randint(0, n - 1)

    def conflicts(self, agents):
        count = 0
        for other in agents:
            if other.row == self.row:
                continue
            if other.col == self.col:
                count += 1
            if abs(other.row - self.row) == abs(other.col - self.col):
                count += 1
        return count

    def move_to_best_position(self, agents):
        min_conflicts = n
        best_cols = []
        for c in range(n):
            self.col = c
            conf = self.conflicts(agents)
            if conf < min_conflicts:
                min_conflicts = conf
                best_cols = [c]
            elif conf == min_conflicts:
                best_cols.append(c)
        self.col = rand.choice(best_cols)


def fast_check(arr, i, j):
    diag1 = np.diagonal(arr, offset=j - i)
    diag2 = np.diagonal(np.fliplr(arr), offset=(arr.shape[1] - j - 1) - i)
    return np.any(diag1 == 1) or np.any(diag2 == 1)


def check_chess_table():
    return np.all(np.any(chess_table, axis=1))


def warmup_numpy():
    dummy = np.zeros((n, n), dtype=int)
    np.diagonal(dummy, offset=1)
    np.fliplr(dummy)
    arr = np.arange(n)
    np.random.shuffle(arr)
    np.random.randint(0, n, size=n)

def board_state(agents):
    board = np.zeros((n, n), dtype=int)
    for q in agents:
        board[q.row, q.col] = 1
    return board


def multi_agents_algo(max_steps=1000):
    agents = [QueenAgent(row=i) for i in range(n)]

    for step in range(max_steps):
        conflicts_list = [q.conflicts(agents) for q in agents]
        total_conflicts = sum(conflicts_list)
        if total_conflicts == 0:
            return board_state(agents), step
        conflicted_agents = [q for q, c in zip(agents, conflicts_list) if c > 0]
        agent = rand.choice(conflicted_agents)
        agent.move_to_best_position(agents)

    return None, max_steps

def recursive_algo(row=0, used_cols=None):
    if used_cols is None:
        used_cols = set()

    if row == n:
        return True

    for col in range(n):
        if col in used_cols or fast_check(chess_table, row, col):
            continue

        chess_table[row, col] = 1
        used_cols.add(col)

        if recursive_algo(row + 1, used_cols):
            return True

        chess_table[row, col] = 0
        used_cols.remove(col)

    return False


def brute_force_algo():
    free_indexes_y = np.arange(n)
    free_indexes_x = np.arange(n)

    max_attempts = 20000
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        chess_table.fill(0)

        np.random.shuffle(free_indexes_y)
        np.random.shuffle(free_indexes_x)

        used_rows = set()
        used_cols = set()

        for i in free_indexes_y:
            for j in free_indexes_x:
                if i in used_rows or j in used_cols:
                    continue
                if fast_check(chess_table, i, j):
                    continue

                chess_table[i, j] = 1
                used_rows.add(i)
                used_cols.add(j)
                break

        if check_chess_table():
            return chess_table.copy()

    return None


def test_by_time(function):
    chess_table.fill(0)
    start_time = time.perf_counter()
    function()
    end_time = time.perf_counter()

    eval_time = end_time - start_time
    # print(f"Время выполнения {end_time - start_time:.6f}s")
    return eval_time

def test_avg_time(function,steps = 100000):
    time_sum =0
    for i in range(steps):
        time_sum += test_by_time(function)
    return time_sum/steps

def main():
    warmup_numpy()
    print(f"Ср. время выполнения: {test_avg_time(brute_force_algo)}")
    print(f"Ср. время выполнения: {test_avg_time(recursive_algo)}")
    print(f"Ср. время выполнения: {test_avg_time(multi_agents_algo)}")


if __name__ == "__main__":
    main()
