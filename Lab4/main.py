import time
import random


def park_miller(seed: int, n: int) -> list:
    # Set constants
    a = 16807
    m = 2_147_483_647
    q = m // a  # 127_773
    r = m % a  # 2836

    # list generate numbers
    numbers = []

    for i in range(n):
        hi = seed // q
        lo = seed - q * hi
        seed = a * lo - r * hi

        if seed < 0:
            seed += m

        numbers.append(seed / m)

    return numbers


def lecuyer(seed1: int, seed2: int, n: int) -> list:
    # Initialize constants
    m1 = 2 ** 31 - 1
    m2 = 2 ** 29 - 1
    a1 = 16807
    a2 = 48271
    q1 = m1 // a1
    q2 = m2 // a2
    r1 = m1 % a1
    r2 = m2 % a2

    # Initialize state
    x1 = seed1 % m1
    x2 = seed2 % m2

    # list generate numbers
    numbers = []

    for i in range(n):
        # Compute new states
        k1 = x1 // q1
        x1 = a1 * (x1 % q1) - r1 * k1
        if x1 < 0:
            x1 += m1

        k2 = x2 // q2
        x2 = a2 * (x2 % q2) - r2 * k2
        if x2 < 0:
            x2 += m2

        # Compute pseudorandom number
        numbers.append((x1 - x2) % m1 / m1)

    return numbers


def lcg(seed: int, n: int) -> int:
    # set constant
    a = 1_103_515_245
    c = 12_345
    m = 2 ** 31

    # list generate numbers
    numbers = []

    for i in range(n):
        seed = (a * seed + c) % m
        numbers.append(seed / m)

    return numbers


def bbs(seed, n):
    p = 24672462467892469787
    q = 396736894567834589803
    # generate number
    numbers = []

    # generate bits
    bits = []

    # Blum integer
    m = p * q

    seed = (seed ** 2) % m

    for _ in range(n):
        for _ in range(random.randint(1, 30)):
            seed = (seed ** 2) % (p * q)
            bit = seed % 2
            bits.append(bit)
        number = int(''.join(map(str, bits)), 2)

        numbers.append(number / int(''.join(['1'] + ['0'] * len(str(number)))))
        bits.clear()
    return numbers


def chi_square(observed, expected_frequency):
    """
    Calculate the chi-square statistic for the observed and expected frequencies.
    """
    chi_sq = sum([((observed[i] - expected_frequency) ** 2) / expected_frequency for i in range(len(observed))])

    return chi_sq


def uniformity_test(sequence, n, num_bins):
    """
    Test the specified pseudo-random number generator using the chi-square test.
    """
    expected_frequency = n / num_bins
    observed_frequency = [0] * num_bins

    for i in range(n):
        bin_index = int(sequence[i] * num_bins)
        observed_frequency[bin_index] += 1

    chi_sq = chi_square(observed_frequency, expected_frequency)
    return chi_sq


def main():
    # set parameters
    seed = int(time.time())
    n = 1_000_000
    num_bins = 100
    alpha = 124.342  # 0.05

    # test Park-Miller generator
    pm_generator = park_miller(seed, n)
    pm_chi_sq = uniformity_test(pm_generator, n, num_bins)
    print(f"Park-Miller chi-square statistic: {pm_chi_sq}")
    if pm_chi_sq <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')

    # test L'Ecuyer generator
    seed2 = 987654321
    lecuyer_generator = lecuyer(seed, seed2, n)
    lecuyer_chi_sq = uniformity_test(lecuyer_generator, n, num_bins)
    print(f"L'Ecuyer chi-square statistic: {lecuyer_chi_sq}")
    if lecuyer_chi_sq <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')

    # test Linear Congruential generator
    lcg_generator = lcg(seed, n)
    lcg_chi_sq = uniformity_test(lcg_generator, n, num_bins)
    print(f"Linear Congruential chi-square statistic: {lcg_chi_sq}")
    if lcg_chi_sq <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')

    # test Blum-Blum-Shub generator
    bbs_generator = bbs(seed, n)
    bbs_chi_sq = uniformity_test(bbs_generator, n, num_bins)
    print(f"Blum-Blum-Shub chi-square statistic: {bbs_chi_sq}")
    if bbs_chi_sq <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (H0 holds true)')


if __name__ == '__main__':
    main()
