import random
import math
import time
import statistics
import matplotlib.pyplot as plt
import numpy as np
import functools

MODE = "function"
FITNESS_FUNCTION = "math.sin(5*x) * math.cos(3*x) + 0.5*math.log(x + 1)"

BIT_LENGTH = 15
REPEATS = 3
X_MIN, X_MAX = 0.0, 1000.0


@functools.lru_cache(maxsize=1)
def get_distribution():
    mu = (X_MIN + X_MAX) / 2
    sigma = (X_MAX - X_MIN) / 2
    return [random.gauss(mu, sigma) for _ in range(1000000)]


def EvaluateIndividualBits(bits, bit_length=BIT_LENGTH, mode=MODE,
                           fitness_function=FITNESS_FUNCTION, distribution=None):
    bstr = ''.join(str(int(b)) for b in bits)
    idx = int(bstr, 2)
    if mode == "function":

        x = X_MIN + (X_MAX - X_MIN) * idx / (2 ** bit_length - 1)
        result = eval(fitness_function, {"math": math}, {"x": x})
        return float(result)
    else:

        dist = distribution if distribution is not None else get_distribution()
        return float(dist[idx])


class GeneticAlgorithm:
    def __init__(self, bit_length=BIT_LENGTH, population_size=20, max_generations=10,
                 cx_prob=0.7, mut_prob=0.1, crossover="onepoint", mutation="flip",
                 selection="tournament", mut_per_bit=True, elite_count=2):
        self.bit_length = bit_length
        self.population_size = population_size
        self.max_generations = max_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.mut_per_bit = mut_per_bit
        self.elite_count = elite_count
        self.population = []
        self.fitness = []
        self.history_best = []
        self.time_spent = 0.0
        self.best_individual = None
        self.best_fitness = None

    def randomBitIndividual(self, n):
        return [random.randint(0, 1) for _ in range(n)]

    def initialize(self):
        self.population = [self.randomBitIndividual(self.bit_length) for _ in range(self.population_size)]
        self.fitness = [EvaluateIndividualBits(ind, self.bit_length) for ind in self.population]

    def tournament_select(self, k=3):
        n = len(self.population)
        if n == 0:
            return self.randomBitIndividual(self.bit_length)
        kk = min(k, n)
        idxs = random.sample(range(n), kk)
        best_idx = max(idxs, key=lambda i: self.fitness[i])
        return list(self.population[best_idx])

    def one_point_crossover(self, a, b):
        if len(a) < 2:
            return list(a), list(b)
        p = random.randint(1, len(a) - 1)
        child1 = a[:p] + b[p:]
        child2 = b[:p] + a[p:]
        return child1, child2

    def uniform_crossover(self, a, b):
        c1, c2 = [], []
        for i in range(len(a)):
            if random.random() < 0.5:
                c1.append(a[i])
                c2.append(b[i])
            else:
                c1.append(b[i])
                c2.append(a[i])
        return c1, c2

    def mutate_flip(self, ind):
        if self.mut_per_bit:
            for i in range(len(ind)):
                if random.random() < self.mut_prob:
                    ind[i] = 1 - ind[i]
        else:
            if random.random() < self.mut_prob:
                pos = random.randrange(0, len(ind))
                ind[pos] = 1 - ind[pos]
        return ind

    def mate(self, a, b):
        if random.random() > self.cx_prob:
            return list(a), list(b)
        if self.crossover == "onepoint":
            return self.one_point_crossover(a, b)
        else:
            return self.uniform_crossover(a, b)

    def run_single(self, seed=None):
        if seed is not None:
            random.seed(seed)
        start = time.time()
        self.initialize()
        if self.fitness:
            self.history_best = [max(self.fitness)]
        for gen in range(self.max_generations):
            newpop = []
            if self.elite_count > 0 and self.population:
                best_idxs = sorted(range(len(self.fitness)), key=lambda i: self.fitness[i], reverse=True)[
                            :self.elite_count]
                for i in best_idxs:
                    newpop.append(list(self.population[i]))
            while len(newpop) < self.population_size:
                p1 = self.tournament_select()
                p2 = self.tournament_select()
                attempts = 0
                while p2 == p1 and attempts < 5:
                    p2 = self.tournament_select()
                    attempts += 1
                c1, c2 = self.mate(p1, p2)
                c1 = self.mutate_flip(c1)
                c2 = self.mutate_flip(c2)
                newpop.append(c1)
                if len(newpop) < self.population_size:
                    newpop.append(c2)
            newfitness = [EvaluateIndividualBits(ind, self.bit_length) for ind in newpop]
            self.population = newpop
            self.fitness = newfitness
            if self.fitness:
                self.history_best.append(max(self.fitness))
            else:
                self.history_best.append(float("-1e20"))
        end = time.time()
        self.time_spent = end - start
        if self.fitness:
            best_idx = max(range(len(self.fitness)), key=lambda i: self.fitness[i])
            self.best_individual = self.population[best_idx]
            self.best_fitness = self.fitness[best_idx]
        return {
            "best_fitness": float(self.best_fitness),
            "best_index": int(''.join(str(b) for b in self.best_individual), 2),
            "time": self.time_spent,
            "history_best": list(self.history_best)
        }


class LinearSearch:
    def __init__(self, bit_length=BIT_LENGTH, mode=MODE, fitness_function=FITNESS_FUNCTION, distribution=None):
        self.bit_length = bit_length
        self.mode = mode
        self.fitness_function = fitness_function
        self.distribution = distribution if distribution is not None else get_distribution()
        self.best_fitness = None
        self.best_index = None
        self.time_spent = 0.0

    def run(self):
        start = time.time()
        best = float("-1e20")
        best_idx = 0
        for idx in range(2 ** self.bit_length):
            bits = [(idx >> j) & 1 for j in reversed(range(self.bit_length))]

            val = EvaluateIndividualBits(bits, self.bit_length, self.mode, self.fitness_function, self.distribution)
            if val > best:
                best = val
                best_idx = idx
        end = time.time()
        self.best_fitness = float(best)
        self.best_index = int(best_idx)
        self.time_spent = end - start
        return {
            "best_fitness": self.best_fitness,
            "best_index": self.best_index,
            "time": self.time_spent
        }


class ExperimentRunner:
    def __init__(self):
        self.results = []
        self.linear_cache = {}

    def run_grid(self, grid, repeats=REPEATS, seed_base=0):
        keys = list(grid.keys())
        values = [grid[k] for k in keys]

        def product(lists):
            if not lists:
                return [[]]
            rest = product(lists[1:])
            res = []
            for item in lists[0]:
                for r in rest:
                    res.append([item] + r)
            return res

        combos = product(values)
        combo_index = 0
        for combo in combos:
            combo_index += 1
            params = dict(zip(keys, combo))
            ga_stats = []
            for r in range(repeats):
                seed = seed_base + combo_index * 100 + r
                ga = GeneticAlgorithm(
                    bit_length=params.get("bit_length", BIT_LENGTH),
                    population_size=params.get("population_size", 20),
                    max_generations=params.get("max_generations", 10),
                    cx_prob=params.get("cxpb", 0.7),
                    mut_prob=params.get("mutpb", 1.0 / BIT_LENGTH),
                    crossover=params.get("crossover", "onepoint"),
                    mutation=params.get("mutation", "flip"),
                    mut_per_bit=params.get("mut_per_bit", True),
                    elite_count=params.get("elite_count", 2),
                )
                res = ga.run_single(seed=seed)
                ga_stats.append(res)
            lin_key = (
                params.get("bit_length", BIT_LENGTH),
                params.get("mode", MODE),
                params.get("fitness_function", FITNESS_FUNCTION)
            )
            if lin_key in self.linear_cache:
                linear_res = self.linear_cache[lin_key]
            else:
                linear = LinearSearch(
                    bit_length=params.get("bit_length", BIT_LENGTH),
                    mode=params.get("mode", MODE),
                    fitness_function=params.get("fitness_function", FITNESS_FUNCTION),
                    distribution=params.get("distribution", get_distribution())
                )
                linear_res = linear.run()
                self.linear_cache[lin_key] = linear_res
            bests = [g["best_fitness"] for g in ga_stats]
            times = [g["time"] for g in ga_stats]
            abs_errors = [abs(b - linear_res["best_fitness"]) for b in bests]
            rel_errors = [ae / (abs(linear_res["best_fitness"]) + 1e-12) for ae in abs_errors]
            histories = [g["history_best"] for g in ga_stats]
            summary = {
                "params": params,
                "linear_best_fitness": linear_res["best_fitness"],
                "linear_time": linear_res["time"],
                "ga_count": len(bests),
                "ga_best_mean": statistics.mean(bests),
                "ga_best_std": statistics.stdev(bests),
                "ga_time_mean": statistics.mean(times),
                "ga_time_min": min(times),
                "accuracy_mean_abs": statistics.mean(abs_errors),
                "accuracy_std_abs": statistics.stdev(abs_errors),
                "accuracy_mean_rel": statistics.mean(rel_errors),
                "accuracy_std_rel": statistics.stdev(rel_errors),
                "ga_runs": ga_stats,
                "ga_histories": histories,
                "linear": linear_res
            }
            self.results.append(summary)
            print(f"Комбинация {combo_index}/{len(combos)} обработана: {params}")
        return self.results

    def print_results(self):
        print("\nИтоги эксперимента\n")
        for i, s in enumerate(self.results, 1):
            p = s["params"]
            print(f"Конфигурация {i}")
            print(
                f"Параметры: crossover={p.get('crossover')}, mutation={p.get('mutation')}, pop={p.get('population_size')}, cxpb={p.get('cxpb')}, mutpb={p.get('mutpb')}, mut_per_bit={p.get('mut_per_bit', True)}")
            print(
                f"Линейный поиск: лучшая приспособленность = {s['linear_best_fitness']:.6f}, время = {s['linear_time']:.4f} с")
            print(f"GA: средняя лучшая приспособленность = {s['ga_best_mean']:.6f}")
            print(f"GA: стандартное отклонение финального результата = {s['ga_best_std']:.6f}")
            print(f"GA: среднее время = {s['ga_time_mean']:.4f} с, минимальное время = {s['ga_time_min']:.4f} с\n")

    def print_top5_mean(self):
        scored = []
        for i, s in enumerate(self.results):
            mean_time = statistics.mean(g["time"] for g in s["ga_runs"])
            mean_val = s["ga_best_mean"]
            linear_best = s["linear_best_fitness"]
            accuracy = 1 - abs(mean_val - linear_best) / (abs(linear_best) + 1e-12)
            scored.append((i + 1, mean_val, mean_time, accuracy, s))
        scored.sort(key=lambda x: -x[1])
        print("\nСписок конфигураций по среднему значению GA\n")
        for rank, (idx, mean_val, mean_time, accuracy, s) in enumerate(scored[:5], 1):
            p = s["params"]
            print(f"{rank}. Конфигурация {idx}: crossover={p.get('crossover')}, mutation={p.get('mutation')}, "
                  f"pop={p.get('population_size')}, cxpb={p.get('cxpb')}, mutpb={p.get('mutpb')}, "
                  f"mut_per_bit={p.get('mut_per_bit', True)} -> mean={mean_val:.6f}, "
                  f"GA accuracy={accuracy:.3f}, mean_time={mean_time:.4f} с")

    def print_top5_best(self):
        scored = []
        for i, s in enumerate(self.results):
            best_val = max(g["best_fitness"] for g in s["ga_runs"])
            mean_time = statistics.mean(g["time"] for g in s["ga_runs"])
            linear_best = s["linear_best_fitness"]
            accuracy = 1 - abs(best_val - linear_best) / (abs(linear_best) + 1e-12)
            scored.append((i + 1, best_val, mean_time, accuracy, s))
        scored.sort(key=lambda x: -x[1])
        print("\nСписок конфигураций по макс значению GA\n")
        for rank, (idx, best_val, mean_time, accuracy, s) in enumerate(scored[:5], 1):
            p = s["params"]
            print(f"{rank}. Конфигурация {idx}: crossover={p.get('crossover')}, mutation={p.get('mutation')}, "
                  f"pop={p.get('population_size')}, cxpb={p.get('cxpb')}, mutpb={p.get('mutpb')}, "
                  f"mut_per_bit={p.get('mut_per_bit', True)} -> best={best_val:.6f}, "
                  f"GA accuracy={accuracy:.3f}, mean_time={mean_time:.4f} с")

    def plot_time_comparison(self, title="Сравнение времени: GA vs Линейный"):
        if not self.results:
            return

        labels, ga_means, linear_times = [], [], []
        for i, s in enumerate(self.results):
            p = s["params"]
            lbl = f"{p.get('crossover')}/{p.get('mutation')}\npop{p.get('population_size')}\ncx{p.get('cxpb')},mu{p.get('mutpb')}"
            labels.append(lbl)
            ga_means.append(s["ga_time_mean"])
            linear_times.append(s["linear_time"])

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        line_ga, = ax.plot(x, ga_means, marker='o', label="GA")
        line_lin, = ax.plot(x, linear_times, marker='s', label="Linear")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel("Время (сек)")
        ax.set_title(title)
        ax.grid(True)

        leg = ax.legend(loc="best", fancybox=True)
        lines = [line_ga, line_lin]
        lined = {}
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)
            lined[legline] = origline

        def on_pick(event):
            legline = event.artist
            origline = lined[legline]
            if event.mouseevent.button == 1:
                origline.set_visible(not origline.get_visible())
            elif event.mouseevent.button == 3:
                for l in lines:
                    l.set_visible(False)
                origline.set_visible(True)
            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)
        plt.tight_layout()
        plt.show()

    def plot_convergence(self, title="Сходимость GA"):
        if not self.results:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        lines = []
        labels = []

        for s in self.results:
            histories = s.get("ga_histories", [])
            if not histories:
                continue
            minlen = min(len(h) for h in histories)
            arr = np.array([h[:minlen] for h in histories])
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            gens = np.arange(minlen)
            lbl = f"{s['params'].get('crossover')}/{s['params'].get('mutation')} pop{s['params'].get('population_size')}"
            line, = ax.plot(gens, mean, linewidth=2, label=lbl)
            ax.fill_between(gens, mean - std, mean + std, alpha=0.15)
            lines.append(line)
            labels.append(lbl)

        ax.set_xlabel("Поколение")
        ax.set_ylabel("Fitness")
        ax.set_title(title)
        ax.grid(True)

        leg = ax.legend(loc="best", fancybox=True)
        lined = {}
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)
            lined[legline] = origline

        def on_pick(event):
            legline = event.artist
            origline = lined[legline]
            if event.mouseevent.button == 1:
                origline.set_visible(not origline.get_visible())
            elif event.mouseevent.button == 3:
                for l in lines:
                    l.set_visible(False)
                origline.set_visible(True)
            fig.canvas.draw()

        fig.canvas.mpl_connect("pick_event", on_pick)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    grid = {
        "bit_length": [BIT_LENGTH],
        "population_size": [100,300],
        "max_generations": [30],
        "cxpb": [0.1, 0.7],
        "mutpb": [round(0.2 / BIT_LENGTH, 3), 0.2],
        "crossover": ["onepoint", "uniform"],
        "mutation": ["flip"],
        "mut_per_bit": [True, False],
        "elite_count": [2, 5]
    }
    runner = ExperimentRunner()
    results = runner.run_grid(grid, repeats=REPEATS, seed_base=42)
    runner.print_results()
    runner.print_top5_best()
    runner.print_top5_mean()
    runner.plot_time_comparison()
    runner.plot_convergence()
