from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for ch in text if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError("vectors must have the same length")
    n = len(predicted)
    if n == 0:
        return 0.0
    return sum((p - e) ** 2 for p, e in zip(predicted, expected)) / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    if number == 1:
        return ""
    result = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power == 1:
            result.append(f"({divisor})")
        elif power > 1:
            result.append(f"({divisor}**{power})")

        divisor += 1
    if number > 1:
        result.append(f"({number})")
    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total + (k + 1) ** 2 <= cube_count:
        k += 1
        total += k * k

    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    s = str(number)
    length = len(s)
    if length < 3:
        return True
    mid = length // 2
    if length % 2 == 1:
        left = s[:mid]
        right = s[mid + 1:]
    else:
        left = s[:mid - 1]
        right = s[mid + 1:]
    return sum(int(ch) for ch in left) == sum(int(ch) for ch in right)
