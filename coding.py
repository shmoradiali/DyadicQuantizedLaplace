from math import ceil, floor, log2


# Signed Coding
def signed_enc(enc, x):
    if x <= 0:
        return enc(-2 * x + 1)
    else:
        return enc(2 * x)


def signed_dec(dec, x):
    y = dec(x)
    if y % 2 == 1:
        return -ceil(y / 2) + 1
    else:
        return y // 2


# Elias Gamma Coding
def elias_gamma_enc(x):
    n = floor(log2(x))
    res = '0' * n
    res += '{0:b}'.format(x)
    return res


def elias_gamma_dec(s):
    n = s.find('1')
    return int(s[n:], 2)


def signed_elias_gamma_enc(x):
    return signed_enc(elias_gamma_enc, x)


def signed_elias_gamma_dec(x):
    return signed_dec(elias_gamma_dec, x)


# Elias Delta Coding
def elias_delta_enc(x):
    n = floor(log2(x))
    res = elias_gamma_enc(n + 1)
    res += '{0:b}'.format(x)[1:]
    return res


def elias_delta_dec(s):
    l = s.find('1')
    n = int(s[l:2 * l + 1], 2) - 1
    return 2**n + int(s[-n:], 2)


def signed_elias_delta_enc(x):
    return signed_enc(elias_delta_enc, x)


def signed_elias_delta_dec(x):
    return signed_dec(elias_delta_dec, x)


# Vanilla Binary
def binary_enc(x):
    return '{0:b}'.format(x)


def binary_dec(s):
    return int(s, 2)


def signed_binary_enc(x):
    return signed_enc(binary_enc, x)


def signed_binary_dec(s):
    return signed_dec(binary_dec, s)
