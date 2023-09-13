from typing import Union
import numpy as np
from numpy import exp
from numpy.random import exponential, random, geometric, laplace
from coding import binary_enc, signed_binary_dec, signed_elias_delta_dec, signed_elias_delta_enc, signed_elias_gamma_dec, signed_elias_gamma_enc, signed_binary_enc
from dql_utils import genT, compute_FT, randchoice, coeff


class DQLMechanism: # Dyadic Quantized Laplace Mechanism
    def __init__(self, eps, l, shared_seed, local_seed, maxt=100, coding_method='elias_gamma'):
        self.shared_rng = np.random.default_rng(seed=shared_seed)
        self.local_rng = np.random.default_rng(seed=local_seed)

        # Coding method
        if coding_method == 'elias_gamma':
            self.to_binary = signed_elias_gamma_enc
            self.from_binary = signed_elias_gamma_dec
        elif coding_method == 'elias_delta':
            self.to_binary = signed_elias_delta_enc
            self.from_binary = signed_elias_delta_dec
        elif coding_method == 'binary': # not a prefix-free code
            self.to_binary = signed_binary_enc
            self.from_binary = signed_binary_dec

        self.l = l # Decoder privacy relaxation parameter
        self.eps = eps # Database privacy parameter
        self.maxt = maxt
        
        self.ds, self.rs, self.Fs = compute_FT(self.l, self.maxt)
 
    def encode(self, x : float, output_binary=False) -> Union[int, str]:
        """
        The encoding function for DQL. If output_binary == True then the function will return
        the binary representation according to the chosen coding method.
        """
        t = genT(self.Fs, self.shared_rng)
        u = self.shared_rng.random() - 0.5

        d = self.ds[t]
        r = self.rs[t]
        c0 = coeff(d)
        c1 = coeff(2*d)
        
        m0, md = randchoice([
            ((0, 2),   1/c0 - r/c1),
            ((-2, -2), (1/c0 - r/c1)*exp(-2*d)),
            ((1, 2),   exp(-d)/c0 - r/c1*(1+exp(-2*d))/2),
            ((-1, -2), exp(-d)/c0 - r/c1*(1+exp(-2*d))/2),
        ], self.local_rng)

        m = m0 + md * (self.local_rng.geometric(1 - exp(-2*d)) - 1)
        m = round(self.eps*x/d + m + (self.local_rng.random() - 0.5) - u)

        if output_binary:
            return self.to_binary(m)
        else:
            return m

    def decode(self, m : Union[str, int], input_binary=False) -> float:
        """
        The decoding function for DQL. If input_binary == True then the function will take
        the binary representation of m as an input.
        """
        t = genT(self.Fs, self.shared_rng)
        u = self.shared_rng.random() - 0.5

        if input_binary:
            m = self.from_binary(m)

        return self.ds[t] * (m + u) / self.eps
