from itertools import count


def main():
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

    for surname, data in surname_count.items():
        if (data['count'] > 1):
            postfix = 5 + data['score'] - 2
        else:
            postfix = 'д'

        print(f"{surname} {postfix}")
    return surname_count


if __name__ == "__main__":
    main()
