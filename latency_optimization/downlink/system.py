from __future__ import annotations

from typing import Any, List

import numpy as np

from latency_optimization.precoders.power import (
    joint_power_scale_numpy,
    normalize_matrix_power_numpy,
)
from latency_optimization.physics.rate_law import RateLaw, resolve_rate_law


class DownlinkSystem:
    """
    Clean downlink simulator:
    - BS transmits with Nb antennas.
    - User k receives with Nr[k] antennas through H[k][l] of shape (Nr[k], Nb[k]).
    - `snr_db` is used only to calibrate isolated noise power.
    - Actual rates use downlink SINR with other users' beams as interference.
    """

    STREAM_H = 0
    STREAM_F = 1

    def __init__(
        self,
        system_params: dict[str, Any],
        seed: int,
        *,
        rate_law: RateLaw | None = None,
    ):
        self.sc = system_params
        self.seed = int(seed)
        self.rate_law = rate_law or resolve_rate_law(str(system_params["finite_blocklength_rate_law"]))

        self.K = int(self.sc["K"])
        self.Nb = np.asarray(self.sc["Nb"], dtype=int)
        self.Nr = np.asarray(self.sc["Nr"], dtype=int)
        self.dk = np.asarray(self.sc["dk"], dtype=int)
        self.B = np.asarray(self.sc["B"], dtype=int)
        self.P = np.asarray(self.sc["P"], dtype=float)
        self.block_power_budget = self._resolve_block_power_budget(self.P)
        self.fs = np.asarray(self.sc["fs"], dtype=float)
        self.snr_db = np.asarray(self.sc["snr_db"], dtype=float)
        self.epsilon = np.asarray(self.sc["epsilon"], dtype=float)
        self.T = np.asarray(self.sc["T"], dtype=int)
        self.initial_bits_per_symbol = np.asarray(self.sc["initial_bits_per_symbol"], dtype=float)
        self.initial_latency = np.asarray(self.sc["initial_latency"], dtype=float)

        self.n_kl: List[List[int]] = [[int(self.T[k])] for k in range(self.K)]
        self.H: List[List[np.ndarray]] = [[] for _ in range(self.K)]
        self.F: List[List[np.ndarray]] = [[] for _ in range(self.K)]
        self.sigma2 = np.full(self.K, np.nan, dtype=float)
        self.R_fbl: List[np.ndarray] = []
        self.C: List[np.ndarray] = []
        self.V: List[np.ndarray] = []
        self.n = np.zeros(self.K, dtype=int)
        self.latency = np.zeros(self.K, dtype=float)

        base_ss = np.random.SeedSequence([self.seed])
        self.base_seed = int(base_ss.generate_state(1, dtype=np.uint32)[0])

        for k in range(self.K):
            self.ensure_block(k, 0)

        self._calibrate_sigma2_from_target_snr()
        self.update_metrics()

    def _rng_for(self, user: int, block: int, stream: int) -> np.random.Generator:
        ss = np.random.SeedSequence([self.base_seed, int(user), int(block), int(stream)])
        return np.random.default_rng(ss)

    @staticmethod
    def _cn(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        real = rng.normal(0.0, 1.0 / np.sqrt(2.0), shape)
        imag = rng.normal(0.0, 1.0 / np.sqrt(2.0), shape)
        return real + 1j * imag

    @staticmethod
    def _normalize_precoder(F: np.ndarray, power: float) -> np.ndarray:
        return normalize_matrix_power_numpy(F, power, zero_fallback=False)

    @staticmethod
    def _resolve_block_power_budget(power_values: np.ndarray) -> float:
        if power_values.size <= 0:
            raise ValueError("Downlink P must contain at least one value.")
        budget = float(power_values[0])
        if not np.allclose(power_values, budget, rtol=1e-6, atol=1e-9):
            raise ValueError(
                "Downlink uses one BS block power budget for the full precoder F_b, "
                "so P must be a scalar or repeated identical values."
            )
        return budget

    def sample_precoder(self, user: int, block: int, variant: int = 0) -> np.ndarray:
        rng_f = self._rng_for(int(user), int(block), self.STREAM_F + int(variant))
        F_kl = self._cn(rng_f, (int(self.Nb[user]), int(self.dk[user])))
        return self._normalize_precoder(F_kl.astype(np.complex128), float(self.block_power_budget))

    def get_block_power(
        self,
        block: int,
        F_override=None,
        active_users: list[int] | None = None,
    ) -> float:
        source = self.F if F_override is None else F_override
        users = [int(k) for k in (active_users if active_users is not None else range(self.K))]
        total_power = 0.0
        for k in users:
            if int(block) >= len(source[k]):
                continue
            Fk = np.asarray(source[k][int(block)], dtype=np.complex128)
            total_power += float(np.linalg.norm(Fk, ord="fro") ** 2)
        return float(total_power)

    def compose_full_precoder(
        self,
        block: int,
        F_override=None,
        active_users: list[int] | None = None,
    ) -> np.ndarray:
        source = self.F if F_override is None else F_override
        users = [int(k) for k in (active_users if active_users is not None else range(self.K))]
        block_beams: list[np.ndarray] = []
        for k in users:
            if int(block) >= len(source[k]):
                continue
            block_beams.append(np.asarray(source[k][int(block)], dtype=np.complex128))
        if len(block_beams) == 0:
            nb = int(self.Nb[0]) if self.K > 0 else 0
            return np.zeros((nb, 0), dtype=np.complex128)
        return np.concatenate(block_beams, axis=1)

    def project_block_precoders_to_power(
        self,
        F_nested,
        block: int,
        active_users: list[int] | None = None,
        eps: float = 1e-12,
    ) -> None:
        users = [int(k) for k in (active_users if active_users is not None else range(self.K))]
        scale = joint_power_scale_numpy(
            (
                np.asarray(F_nested[k][int(block)], dtype=np.complex128)
                for k in users
                if int(block) < len(F_nested[k])
            ),
            self.block_power_budget,
            eps=eps,
        )
        if scale is None:
            return
        for k in users:
            if int(block) >= len(F_nested[k]):
                continue
            F_nested[k][int(block)] = np.asarray(F_nested[k][int(block)], dtype=np.complex128) * scale

    def ensure_block(self, user: int, block: int, template_precoder: np.ndarray | None = None) -> None:
        k = int(user)
        l = int(block)

        while len(self.H[k]) <= l:
            rng_h = self._rng_for(k, len(self.H[k]), self.STREAM_H)
            H_kl = self._cn(rng_h, (int(self.Nr[k]), int(self.Nb[k])))
            self.H[k].append(H_kl.astype(np.complex128))

        while len(self.F[k]) <= l:
            if template_precoder is not None:
                F_kl = np.array(template_precoder, dtype=np.complex128, copy=True)
            else:
                F_kl = self.sample_precoder(k, len(self.F[k]))
            self.F[k].append(np.asarray(F_kl, dtype=np.complex128))
            self.project_block_precoders_to_power(self.F, len(self.F[k]) - 1)

    def clone_precoders(self) -> List[List[np.ndarray]]:
        return [[np.array(F_kl, copy=True) for F_kl in user_F] for user_F in self.F]

    def _resolve_block_index(self, user: int, block: int, F_override=None) -> int:
        source = self.F if F_override is None else F_override
        if len(source[user]) == 0:
            raise ValueError(f"No precoders available for user {user}")
        l = int(block)
        if l >= len(source[user]):
            raise ValueError(f"User {user} has no block {l}")
        return l

    def _signal_power_per_rx(self, user: int, block: int, F_override=None) -> float:
        k = int(user)
        l = self._resolve_block_index(k, block, F_override=F_override)
        Hk = self.H[k][l]
        Fk = self.F[k][l] if F_override is None else np.asarray(F_override[k][l], dtype=np.complex128)
        HF = Hk @ Fk
        return float(np.linalg.norm(HF, ord="fro") ** 2 / max(1, int(self.Nr[k])))

    def _calibrate_sigma2_from_target_snr(self) -> None:
        for k in range(self.K):
            p_sig = self._signal_power_per_rx(k, 0)
            snr_lin = 10.0 ** (float(self.snr_db[k]) / 10.0)
            self.sigma2[k] = p_sig / max(snr_lin, 1e-30)

    def get_interference_plus_noise_covariance(self, user: int, block: int, F_override=None) -> np.ndarray:
        k = int(user)
        l = self._resolve_block_index(k, block, F_override=F_override)
        Hk = self.H[k][l]
        Nrk = int(self.Nr[k])
        cov = float(self.sigma2[k]) * np.eye(Nrk, dtype=np.complex128)

        source = self.F if F_override is None else F_override
        for j in range(self.K):
            if j == k or int(block) >= len(source[j]):
                continue
            Fj = np.asarray(source[j][int(block)], dtype=np.complex128)
            HFj = Hk @ Fj
            cov += HFj @ HFj.conj().T

        cov = 0.5 * (cov + cov.conj().T)
        cov += 1e-9 * np.eye(Nrk, dtype=np.complex128)
        return cov

    def compute_block_rate(self, user: int, block: int, n_kl: int, F_override=None) -> float:
        k = int(user)
        l = self._resolve_block_index(k, block, F_override=F_override)
        Hk = self.H[k][l]
        Fk = self.F[k][l] if F_override is None else np.asarray(F_override[k][l], dtype=np.complex128)
        noise_cov = self.get_interference_plus_noise_covariance(k, l, F_override=F_override)
        return self.rate_law.mimo_numpy(
            Hk,
            Fk,
            noise_cov,
            int(n_kl),
            float(self.epsilon[k]),
        ).rate

    def apply_solution(self, F_new: List[List[np.ndarray]], n_kl_new: List[List[int]]) -> None:
        self.n_kl = [list(map(int, blocks)) for blocks in n_kl_new]
        for k in range(self.K):
            for l, n_kl in enumerate(self.n_kl[k]):
                if int(n_kl) <= 0:
                    raise ValueError(
                        f"Downlink n_kl must be strictly positive; got n_kl[{k}][{l}]={int(n_kl)}."
                    )
        for k in range(self.K):
            for l in range(len(self.n_kl[k])):
                template = None
                if k < len(F_new) and l < len(F_new[k]):
                    template = np.asarray(F_new[k][l], dtype=np.complex128)
                elif len(self.F[k]) > 0:
                    template = np.asarray(self.F[k][-1], dtype=np.complex128)
                self.ensure_block(k, l, template_precoder=template)

            self.H[k] = self.H[k][: len(self.n_kl[k])]
            self.F[k] = self.F[k][: len(self.n_kl[k])]

        for k in range(self.K):
            for l in range(len(self.n_kl[k])):
                self.F[k][l] = np.asarray(F_new[k][l], dtype=np.complex128)

        max_blocks = max((len(blocks) for blocks in self.n_kl), default=0)
        for l in range(max_blocks):
            self.project_block_precoders_to_power(self.F, l)

        self.update_metrics()

    def update_metrics(self) -> None:
        self.n = np.array([sum(v) for v in self.n_kl], dtype=int)
        self.latency = self.n / self.fs

        self.C = []
        self.V = []
        self.R_fbl = []
        for k in range(self.K):
            Ck = []
            Vk = []
            Rk = []
            for l, n_kl in enumerate(self.n_kl[k]):
                Hk = self.H[k][l]
                Fk = self.F[k][l]
                noise_cov = self.get_interference_plus_noise_covariance(k, l)
                rate_result = self.rate_law.mimo_numpy(
                    Hk,
                    Fk,
                    noise_cov,
                    int(n_kl),
                    float(self.epsilon[k]),
                )
                Ck.append(rate_result.capacity)
                Vk.append(rate_result.dispersion)
                Rk.append(rate_result.rate)
            self.C.append(np.asarray(Ck, dtype=float))
            self.V.append(np.asarray(Vk, dtype=float))
            self.R_fbl.append(np.asarray(Rk, dtype=float))

    def get_snr_sinr_db(self) -> tuple[list[float], list[float]]:
        snr_db = []
        sinr_db = []
        for k in range(self.K):
            sig_vals = []
            int_vals = []
            for l in range(len(self.n_kl[k])):
                Hk = self.H[k][l]
                Fk = self.F[k][l]
                desired_power = float(np.linalg.norm(Hk @ Fk, ord="fro") ** 2 / max(1, int(self.Nr[k])))
                interference_power = 0.0
                for j in range(self.K):
                    if j == k or l >= len(self.F[j]):
                        continue
                    Fj = self.F[j][l]
                    interference_power += float(
                        np.linalg.norm(Hk @ Fj, ord="fro") ** 2 / max(1, int(self.Nr[k]))
                    )
                sig_vals.append(desired_power)
                int_vals.append(interference_power)
            mean_sig = float(np.mean(sig_vals)) if sig_vals else 0.0
            mean_int = float(np.mean(int_vals)) if int_vals else 0.0
            mean_noise = float(self.sigma2[k])
            snr_db.append(10.0 * np.log10(max(mean_sig / max(mean_noise, 1e-30), 1e-30)))
            sinr_db.append(10.0 * np.log10(max(mean_sig / max(mean_int + mean_noise, 1e-30), 1e-30)))
        return snr_db, sinr_db
