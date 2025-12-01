from itertools import count
import io
import sys
import time

def my_own_realisation():
    with open('olymp1.txt') as f1, open('olymp2.txt') as f2:
        lines = f1.readlines() + f2.readlines()

    surname_count = {}
    for line in lines:
        split_line = line.split()
        surname = split_line[0]

        if surname not in surname_count:
            surname_count[surname] = {'score': 1 if len(split_line) >= 2 else 0, 'count': 1}
            continue

        surname_count[surname]['score'] += 1 if len(split_line) >= 2 else 0
        surname_count[surname]['count'] += 1

    sorted_surnames = sorted(surname_count.items())
    for surname, data in sorted_surnames:
        postfix = 5 + data['score'] - 2 if data['count'] > 1 else 'д'
        print(f"{surname} {postfix}")

def grok_realisation():
    d1={};[d1.update({p[0]:d1.get(p[0],False)or len(p)>1}) for l in open('olymp1.txt') if (p:=l.strip().split())]
    d2={};[d2.update({p[0]:d2.get(p[0],False)or len(p)>1}) for l in open('olymp2.txt') if (p:=l.strip().split())]
    for s in sorted(set(d1)|set(d2)):print(s,'д'if s not in d1 or s not in d2 else'5'if d1[s]and d2[s]else'4'if d1[s]or d2[s]else'3')

def chatgpt_realisation():
    def read(p):
        d={}
        for l in open(p, encoding='utf-8'):
            s=l.split()
            if not s: continue
            n=s[0]
            win = len(s)>1 and s[1] in '123'
            d[n] = d.get(n, False) or win
        return d

    a, b = read('olymp1.txt'), read('olymp2.txt')
    for n in sorted(set(a) | set(b)):
        if (n in a) ^ (n in b):
            print(n, 'д')
        else:
            print(n, '345'[(a.get(n, False) + b.get(n, False))])


def deepseek_realisation():
    d1, d2 = {}, {}
    with open('olymp1.txt') as f1, open('olymp2.txt') as f2:
        for f, d in [(f1, d1), (f2, d2)]:
            for line in f:
                parts = line.split()
                name, prize = parts[0], len(parts) > 1
                d[name] = d.get(name, False) or prize

    all_names = set(d1) | set(d2)
    for name in sorted(all_names):
        p1, p2 = name in d1, name in d2
        pr1, pr2 = d1.get(name, False), d2.get(name, False)
        if p1 + p2 == 1:
            print(name, 'д')
        else:
            print(name, [3, 4, 5][pr1 + pr2])


def qwen_realisation():
    students = {}
    for fname in ['olymp1.txt', 'olymp2.txt']:
        with open(fname, encoding='utf-8') as f:
            for line in f:
                if not (line := line.strip()): continue
                parts = line.split()
                surname = parts[0]
                prize = len(parts) > 1 and parts[1] in {'1', '2', '3'}
                entry = students.setdefault(surname, [0, 0])
                entry[0] += 1
                entry[1] += prize
    for surname in sorted(students):
        part, prizes = students[surname]
        res = 'д' if part == 1 else '5' if prizes == 2 else '4' if prizes == 1 else '3'
        print(surname, res)

def capture_output(func):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        func()
        return sys.stdout.getvalue().strip().split("\n")
    finally:
        sys.stdout = old_stdout



def compare_all(runs=100):
    funcs = [
        ("my_own_realisation", my_own_realisation),
        ("grok_realisation", grok_realisation),
        ("chatgpt_realisation", chatgpt_realisation),
        ("deepseek_realisation", deepseek_realisation),
        ("qwen_realisation", qwen_realisation),
    ]

    results = {}
    avg_timings = {}

    print(f"Сравнение реализаций на {runs} прогонов\n")

    for name, f in funcs:
        print(f"Запуск {name}...")
        start = time.perf_counter()
        for i in range(1, runs + 1):
            capture_output(f)
            if i % (runs // 100) == 0:
                progress = i / runs * 100
                print(f"\r{name}: {progress:.1f}% выполнено", end='')
                sys.stdout.flush()
        end = time.perf_counter()
        avg_time = (end - start) / runs
        avg_timings[name] = avg_time
        results[name] = capture_output(f)
        print(f"\r{name}: 100.0% выполнено")

    names = list(results.keys())
    print("\nСравнение вывода функций:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            print(f"{n1} vs {n2}: {'СОВПАДАЮТ' if results[n1] == results[n2] else 'ОТЛИЧАЮТСЯ'}")

    print("\nСреднее время выполнения (сек):")
    for name in names:
        print(f"{name:22s} {avg_timings[name]:.10f}")

    print("\nСортировка по скорости (от самой быстрой):")
    for name, t in sorted(avg_timings.items(), key=lambda x: x[1]):
        print(f"{name:22s} {t:.10f}")

    print("\nДетальный анализ различий\n")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            if results[n1] == results[n2]:
                continue
            print(f"\nРазличия между {n1} и {n2}:")
            r1, r2 = results[n1], results[n2]
            max_len = max(len(r1), len(r2))
            for k in range(max_len):
                line1 = r1[k] if k < len(r1) else "<нет строки>"
                line2 = r2[k] if k < len(r2) else "<нет строки>"
                if line1 != line2:
                    print(f"- Строка {k+1}:")
                    print(f"  {n1}: {line1}")
                    print(f"  {n2}: {line2}")


def main():
    compare_all()


if __name__ == "__main__":
    main()
