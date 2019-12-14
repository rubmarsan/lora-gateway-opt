import bisect
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from math import pow, ceil
import pickle
import numpy as np
from scipy import optimize


class LoRaWorld:
    def __init__(self, lambdas_, lengths, tx_priorities, snrs, c_opt, alpha, model_paths, acceptable_tx_powers, forced_configs = None, forced_nodes = None):
        """
        :type lambdas_ np.ndarray
        :param lambdas_: generation rate (packets per second)
        :param lengths: lengths, IN BYTES, of the payloads
        :param tx_priorities: priorities of nodes (natural numbers)
        :param snrs: SNRs of each node € (-inf, inf)
        :param c_opt: Matrix that indicates the best config for each node (49 x lambdas_.shape[0])
        :param alpha: Parameter to adjust the importance of Throughput vs Current Consumption
        :param model_paths: Path to the pickle file where the PRR model is specified
        """
        assert lambdas_.shape[0] == lengths.shape[0] == tx_priorities.shape[0], 'Incorrect input size'
        assert acceptable_tx_powers.shape[0] == 4, 'Number of tx_power considered does not match'
        assert acceptable_tx_powers.shape[1] == lambdas_.shape[0]
        assert len(model_paths) == lambdas_.shape[0]

        self.lambdas_ = np.array(lambdas_)
        self.lengths = np.array(lengths)
        self.tx_priorities = np.array(tx_priorities)
        self.alpha = alpha
        self.snrs = np.array(snrs)
        self.N = lambdas_.shape[0]
        self.C = np.zeros((49, self.N))  # :type np.ndarray
        self.C_opt = np.array(c_opt)  # :type np.ndarray
        self.R = 0
        self.BW = 125e3
        self.preamble_symbols = 8
        self.header_length = 13
        self.explicit_header = 1
        # warnings.warn('cambiar aqui')
        self.DC = 0.01  # 0.1%
        # warnings.warn('cambiar aqui')
        self.TDC_MAX = 1000  # 3600 * self.DC
        self.last_packet = None
        self.starting_config = None
        self.ToA = None
        self.factor_l = None
        self.prr_model = [pickle.load(open(model_path, 'rb')) for model_path in model_paths]
        self.acceptable_tx_powers = acceptable_tx_powers

        if forced_nodes is None:
            self.forced_nodes = []
            self.forced_configs = None
        else:
            assert 0 <= len(forced_nodes) <= self.N
            self.forced_nodes = forced_nodes
            assert forced_configs.shape == self.C.shape
            self.forced_configs = forced_configs

        # for faster gym
        self.future_events = None
        self.time_line = None
        self.t = None
        #

        # for TDC
        self.last_TDC = None
        #

        self.min_lambda = None
        self.max_lambda = None

        self.reporting_consumption = None
        self.min_consumption = None
        self.max_consumption = None

        self.min_tx_priority = None
        self.max_tx_priority = None

        self.min_power_priority = None
        self.max_power_priority = None

        self.alphas = np.array(
            [0, -30.2580, -77.1002, -244.6424, -725.9556, -2109.8064, -4452.3653, -105.1966, -289.8133, -1114.3312,
             -4285.4440, -20771.6945, -98658.1166])
        self.betas = np.array(
            [0, 0.2857, 0.2993, 0.3223, 0.3340, 0.3407, 0.3317, 0.3746, 0.3756, 0.3969, 0.4116, 0.4332, 0.4485])

        self.ToA = np.zeros([49, self.N])
        for c in range(1, 13):
            for nodo in range(self.N):
                self.ToA[[c, c + 12, c + 24, c + 36], nodo] = self.compute_over_the_air_time(self.lengths[nodo],
                                                                                             *self.compute_sf_cr(c))

        nerfs = np.array(   # difference in SNR when power is not the highest one (note that max power is 13, not 14)
            [-11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
             -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7,
             -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

        nerfs = np.tile(nerfs, (self.N, 1)).T
        self.snrs_nerfed = (nerfs + self.snrs)
        self.snrs_threshold = self.snrs_nerfed + 6

        self.factor_l = np.zeros([49, self.N])
        for c in range(1, 13):
            for txp_i, txp in enumerate([2, 6, 10, 14]):
                for nodo in range(self.N):
                    acceptable_consumption = self.acceptable_tx_powers[txp_i, nodo]
                    prr = self.compute_prr(self.lengths[nodo], *self.compute_sf_cr(c), txp, self.prr_model[nodo])
                    length = self.lengths[nodo]
                    priority = self.tx_priorities[nodo]
                    lambda_ = self.lambdas_[nodo]

                    self.factor_l[c + (txp_i * 12), nodo] = prr * length * priority * lambda_ * acceptable_consumption

        # compute_consumption(self, payload_length, action, pot_tx):
        self.current_consumption = np.zeros([49, self.N])
        for c in range(1, 13):
            for txp_i, txp in enumerate([2, 6, 10, 14]):
                for nodo in range(self.N):
                    self.current_consumption[c + (txp_i * 12), nodo] = self.compute_consumption(self.lengths[nodo], c,
                                                                                                txp)

                    # print('Gym LoRa initialization done')

    # poner los breakpoints dentro de funciones (es decir: no en el main)
    def get_reward_matricial_alt(self, X):
        C = X.reshape(49, self.N)
        # lambdas_agg = self.lambdas_.dot(C.T)
        # lambdas_agg_rel = np.tile(lambdas_agg, (self.N, 1)).T - (C * self.lambdas_)
        # lambdas_agg_rel_folded = (lambdas_agg_rel[1:7, :] + lambdas_agg_rel[7:13, :] + lambdas_agg_rel[13:19, :]
        #                           + lambdas_agg_rel[19:25, :] + lambdas_agg_rel[25:31, :] + lambdas_agg_rel[31:37, :]
        #                           + lambdas_agg_rel[37:43, :] + lambdas_agg_rel[43:49, :])
        # lambdas_any = np.vstack([np.zeros((1, self.N)), np.tile(lambdas_agg_rel_folded, (8, 1))])
        # # lambdas_agg_rel_folded_stacked <- cualquier nodo ha transmitido en el mismo SF

        lambdas_any = np.zeros((49, self.N))
        for c in range(1, 49):
            for n in range(self.N):
                n_c = (c - 1) % 6  # SF
                config_filter = np.zeros((48, self.N))
                config_filter[[n_c, n_c + 6, n_c + 12, n_c + 18, n_c + 24, n_c + 30, n_c + 36, n_c + 42], :] = 1
                config_filter[:, n] = 0
                lambdas_any[c, n] = (config_filter * C[1:, :] * self.lambdas_ * self.ToA[1:, :]).sum()

        lambdas_higher = np.zeros((49, self.N))
        for c in range(1, 49):
            for n in range(self.N):
                n_c = (c - 1) % 6  # SF
                matrix_filter = (self.snrs_threshold >= self.snrs_nerfed[c - 1, n]).astype(np.int)
                matrix_filter[:, n] = 0
                config_filter = np.zeros((48, self.N))
                config_filter[[n_c, n_c + 6, n_c + 12, n_c + 18, n_c + 24, n_c + 30, n_c + 36, n_c + 42], :] = 1
                lambdas_higher[c, n] = (matrix_filter * config_filter * C[1:, :] * self.lambdas_).sum()

        lambdas_higher *= self.ToA

        lambdas_tot = lambdas_any + lambdas_higher

        R_T = (self.factor_l * C * np.exp(-lambdas_tot)).sum()
        R_C = (self.current_consumption * C * self.lambdas_).sum()
        R = self.alpha * R_T - (1 - self.alpha) * R_C

        return -R / self.N

    def get_positive_reward_matricial(self, X):
        return -self.get_reward_matricial(X)

    def get_reward_matricial_13(self, X):
        C = X.reshape(13, self.N)
        Y = np.zeros((49, self.N))
        Y[0, :] = C[0, :]
        Y[1 + 12 * 3: 1 + 12 * 4] = C[1:13, :]
        return self.get_reward_matricial(Y)

    def get_reward_matricial(self, X):
        # global N, lengths, lambdas_, priorities, SNRs, ToA, factor_l
        C = X.reshape(49, self.N)


        # la idea aquí es forzar algunos valores de C
        if self.forced_nodes is not None:
            for node in self.forced_nodes:
                C[:, node] = self.forced_configs[:, node]


        lambdas_agg = self.lambdas_.dot(C.T)  # self.lambdas_[:self.N].dot(C.T)

        lambdas_agg_folded = np.hstack([lambdas_agg[1:7] + lambdas_agg[7:13] + lambdas_agg[13:19] + lambdas_agg[
                                                                                                    19:25] + lambdas_agg[
                                                                                                             25:31] + lambdas_agg[
                                                                                                                      31:37] + lambdas_agg[
                                                                                                                               37:43] + lambdas_agg[
                                                                                                                                        43:49],
                                        lambdas_agg[1:7] + lambdas_agg[7:13] + lambdas_agg[13:19] + lambdas_agg[
                                                                                                    19:25] + lambdas_agg[
                                                                                                             25:31] + lambdas_agg[
                                                                                                                      31:37] + lambdas_agg[
                                                                                                                               37:43] + lambdas_agg[
                                                                                                                                        43:49]])
        # lambdas_agg_folded = np.tile(lambdas_agg[1:].reshape(8, 6).sum(axis=0), 2)
        exp_factor = -2 * self.ToA * np.vstack([np.zeros((1, self.N)), np.tile(lambdas_agg_folded, (self.N, 4)).T])

        R_T = (self.factor_l * C * np.exp(exp_factor)).sum()
        if self.alpha == 1:
            R_C = 0
        else:
            R_C = (self.current_consumption * C * self.lambdas_).sum()

        R = self.alpha * R_T - (1 - self.alpha) * R_C

        if self.forced_nodes is not None:
            return -R / (self.N - len(self.forced_nodes))
        else:
            return -R / self.N

    def find_optimal_by_adr(self):
        """
    If NStep > 0 the data rate can be increased and/or power reduced. If Nstep < 0, power can be increased (to the max.).

    For NStep > 0, first the data rate is increased (by Nstep) until DR5 is reached. If the number of steps < Nstep, the remainder is used to decrease the TXpower by 3dBm per step, until TXmin is reached. TXmin = 2 dBm for EU868.
        """

        # SNRmargin = SNRm – RequiredSNR(DR) - margin_db

        # simulate 20 readings in an std=4 LNSPL model
        SNRs_max = self.snrs + np.random.normal(0, 4, (20, self.N)).max(axis=0)
        for nodo in range(self.N):
            config = self.get_conf_by_adr(SNRs_max[nodo])
            self.C_opt[:, nodo] = np.zeros(49)
            self.C_opt[config, nodo] = 1

        return self.C_opt, None

    def find_optimal_c(self, passes=1, verbose=True):
        assert 0 < passes < 100

        constraints = list()
        for nodo in range(self.N):
            constraints.append(
                {
                    'type': 'eq',
                    'fun': lambda x, nodo=nodo: x[nodo + self.N * 0] + x[nodo + self.N * 1] + x[nodo + self.N * 2] + x[
                        nodo + self.N * 3] + x[nodo + self.N * 4] + x[nodo + self.N * 5] + x[nodo + self.N * 6] + x[
                                                    nodo + self.N * 7] + x[nodo + self.N * 8] + x[nodo + self.N * 9] +
                                                x[nodo + self.N * 10] + x[nodo + self.N * 11] + x[nodo + self.N * 12] +
                                                x[nodo + self.N * 13] + x[nodo + self.N * 14] + x[nodo + self.N * 15] +
                                                x[nodo + self.N * 16] + x[nodo + self.N * 17] + x[nodo + self.N * 18] +
                                                x[nodo + self.N * 19] + x[nodo + self.N * 20] + x[nodo + self.N * 21] +
                                                x[nodo + self.N * 22] + x[nodo + self.N * 23] + x[nodo + self.N * 24] +
                                                x[nodo + self.N * 25] + x[nodo + self.N * 26] + x[nodo + self.N * 27] +
                                                x[nodo + self.N * 28] + x[nodo + self.N * 29] + x[nodo + self.N * 30] +
                                                x[nodo + self.N * 31] + x[nodo + self.N * 32] + x[nodo + self.N * 33] +
                                                x[nodo + self.N * 34] + x[nodo + self.N * 35] + x[nodo + self.N * 36] +
                                                x[nodo + self.N * 37] + x[nodo + self.N * 38] + x[nodo + self.N * 39] +
                                                x[nodo + self.N * 40] + x[nodo + self.N * 41] + x[nodo + self.N * 42] +
                                                x[nodo + self.N * 43] + x[nodo + self.N * 44] + x[nodo + self.N * 45] +
                                                x[nodo + self.N * 46] + x[nodo + self.N * 47] + x[
                                                    nodo + self.N * 48] - 1
                }
            )

        bounds = list()
        for x in range(self.N * 49):
            bounds.append((0, 1))

        # options = {'disp': False, 'maxiter': 1e6, 'ftol': 1e-3, 'iprint': 0}
        if verbose:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-5, 'iprint': 2}
        else:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-5, 'iprint': 0}

        max_R = -float('inf')
        for _ in range(int(passes)):
            initial_value = np.random.rand(49 * self.N)
            initial_value.shape = (49, self.N)
            for nodo in range(self.N):
                initial_value[:, nodo] /= initial_value[:, nodo].sum()

            res = optimize.minimize(self.get_reward_matricial, initial_value, method='SLSQP',
                                    bounds=bounds, constraints=constraints, options=options)

            if (-res.fun) > max_R:
                max_R = -res.fun
                self.C_opt = res.x.reshape(49, self.N)
                if verbose:
                    print('New max found')

        assert max_R > 0, 'Debe ser mayor que 0'
        return self.C_opt, max_R

    def map_solutions(self, solutions, population):
        solutions.shape = (population, 13, self.N)
        solutions = [(solutions[_] - solutions[_].min(axis=0)) / solutions[_].max(axis=0) for _ in range(population)]
        solutions = [solutions[_] / solutions[_].sum(axis=0) for _ in range(population)]
        return solutions

    def find_optimal_c_13(self, passes=1, verbose=True, initial=None):
        assert 0 < passes < 100

        constraints = list()
        for nodo in range(self.N):
            constraints.append(
                {
                    'type': 'eq',
                    'fun': lambda x, nodo=nodo: x[nodo + self.N * 0] + x[nodo + self.N * 1] + x[nodo + self.N * 2] + x[
                        nodo + self.N * 3] + x[nodo + self.N * 4] + x[nodo + self.N * 5] + x[nodo + self.N * 6] + x[
                                                    nodo + self.N * 7] + x[nodo + self.N * 8] + x[nodo + self.N * 9] +
                                                x[nodo + self.N * 10] + x[nodo + self.N * 11] + x[
                                                    nodo + self.N * 12] - 1
                }
            )

        bounds = list()
        for x in range(self.N * 13):
            bounds.append((0, 1))

        # options = {'disp': False, 'maxiter': 1e6, 'ftol': 1e-3, 'iprint': 0}
        if verbose:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-4, 'iprint': 2}
        else:
            options = {'disp': True, 'maxiter': 1e4, 'ftol': 1e-4, 'iprint': 0}

        max_R = -float('inf')
        for _ in range(int(passes)):
            # initial_value = self.find_optimal_by_adr()[0][1 + 12 * 3: 1 + 12 * 4, :]
            if initial is None:
                initial_value = np.random.rand(13 * self.N)
                initial_value.shape = (13, self.N)
            else:
                if initial.shape[0] > 13:
                    initial_value = np.zeros((13, self.N))
                    initial_value[0, :] = initial[0, :]
                    initial_value[1:13, :] = initial[1 + 12 * 3:1 + 12 * 4]
                else:
                    initial_value = initial

            for nodo in range(self.N):
                initial_value[:, nodo] /= initial_value[:, nodo].sum()

            res = optimize.minimize(self.get_reward_matricial_13, initial_value, method='SLSQP',
                                    bounds=bounds, constraints=constraints, options=options)

            if (-res.fun) > max_R:
                max_R = -res.fun
                # aqui meter ya los ceros

                C = res.x.reshape(13, self.N)
                Y = np.zeros((49, self.N))
                Y[0, :] = C[0, :]
                Y[1 + 12 * 3: 1 + 12 * 4, :] = C[1:13, :]

                self.C_opt = Y
                print('New max found')

        assert max_R > 0, 'Debe ser mayor que 0'
        return self.C_opt, max_R

    @staticmethod
    def compute_current_drawn(over_the_air_time, pot_tx):
        current_consumption = {
            2: 76.01,  # 2
            3: 78.27,  # 3
            4: 80.59,  # 4
            5: 83.75,  # 5
            6: 85.53,  # 6
            7: 89.02,  # 7
            8: 93.20,  # 8
            9: 94.14,  # 9
            10: 101.35,  # 10
            11: 103.32,  # 11
            12: 106.54,  # 12
            13: 114.12,  # 13
            14: 114.15,  # 14
        }

        assert pot_tx in current_consumption
        draw_ma = current_consumption[pot_tx]
        return over_the_air_time * draw_ma

    def compute_consumption(self, payload_length, action, pot_tx):
        """
        Computes current consumption derived from transmitting the packet (in mA)
        :param payload_length: Length of the transmitted packet in bytes (only the payload)
        :param action: Action carried out, from 0 to 12
        :param pot_tx: Transmitting power € [2, 14]
        :return:
        """
        sf, cr = self.compute_sf_cr(action)
        ToA = self.compute_over_the_air_time(payload_length, sf, cr)
        return self.compute_current_drawn(ToA, pot_tx)

    @staticmethod
    def compute_sf_cr(action):
        return np.array([0, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12])[action], \
               np.array([0, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7])[action]

    def compute_prr(self, packet_length, sf, cr, tx_power, prr_model):
        # packet_length in bytes
        if sf == 0 and cr == 0:
            return 0

        assert 1 <= packet_length <= 1000
        assert cr in (5, 7), "CR not implemented yet"
        assert 7 <= sf <= 12, "Invalid SF"
        assert tx_power in range(2, 15)

        # warnings.warn('Quitad estas 3 lineas en el futuro')
        # cr = cr if cr == 5 else 8
        # if np.isnan(prr_model[sf, cr - 4, tx_power]):
        #     return 0

        warnings.warn('Asumiendo packet length == 17')
        return prr_model[sf, cr - 4, tx_power] ** (packet_length / 17)

        # action = (sf - 6) + [0, 6][cr == 7]
        # alfa = self.alphas[action]
        # beta = self.betas[action]
        # ber = np.power(10, alfa * np.exp(beta * snr))
        # return np.power(1 - ber, packet_length * 8)

    @staticmethod
    def get_conf_by_adr(snr):
        # returns the config as an index from 1 to 12

        required_SNR = {7: 2.5,  # this are the required SNR values for each SF plus a 10dB margin
                        8: 0,  # see https://github.com/TheThingsNetwork/ttn/issues/265
                        9: -2.5,
                        10: -5,
                        11: -7.5,
                        12: -10
                        }

        best = 12
        for sf, sf_margin in required_SNR.items():
            if sf_margin < snr:
                best = sf
                break

        chosen_power_index = 3
        power_margins = (6, 5.6, 6.5)  # the transmission Power we lose in each TXPOWER step we decrease
        if best == 7:
            gross_margin = snr - 2.5
            for pm in power_margins:
                if gross_margin > pm:
                    gross_margin -= pm
                    chosen_power_index -= 1
                else:
                    break

        CR = 0  # 0 -> CR=4/5 and 1 -> CR=4/7

        best -= 6
        best += (6 * CR)
        best += (12 * chosen_power_index)  # max power

        return best

    def compute_over_the_air_time(self, payload_length, sf, cr, continuous=False):
        # payload_length in bytes

        if sf == 0 and cr == 0:
            return 0

        assert 7 <= sf <= 12
        assert 5 <= cr <= 7
        de = 1 if sf >= 11 else 0
        # http://forum.thethingsnetwork.org/t/spreadsheet-for-lora-airtime-calculation/1190/15
        t_sym = pow(2, sf) / self.BW * 1000  # symbol time in ms
        t_preamble = (self.preamble_symbols + 4.25) * t_sym  # over the air time of the preamble
        if continuous:
            payload_symbol_number = 8 + (((8 * (payload_length + self.header_length) - 4 * sf + 28 + 16 - 20 * (
                1 - self.explicit_header)) / (4 * (sf - 2 * de))) * cr)
        else:
            payload_symbol_number = 8 + max([(ceil(
                (8 * (payload_length + self.header_length) - 4 * sf + 28 + 16 - 20 * (1 - self.explicit_header)) / (
                    4 * (sf - 2 * de))) * cr), 0])  # number of symbols of the payload

        t_payload = payload_symbol_number * t_sym  # payload time in ms
        t_packet = t_preamble + t_payload

        return t_packet / 1000  # expressed in seconds

    def get_transmittable(self, node):
        return 1
        # d_t = 1e-3
        # lambda_ = self.lambdas_[node]
        # length = self.lengths[node]
        # config = self.C[:, node]
        # transmittables_factors = np.array(
        #     [((self.compute_over_the_air_time(length, *self.compute_sf_cr(action)) / self.DC / d_t) - 1) * (
        #         1 - exp(-lambda_ * config[action] * d_t)) for action in np.arange(13)])
        # transmittables_factors[0] = 1
        #
        # return 1 / transmittables_factors.sum()

    def get_effective_lambda(self, node):
        lambda_ = self.lambdas_[node]
        return lambda_ * self.get_transmittable(node)

    def compute_network_performance(self):
        return -self.get_reward_matricial(self.C)

    def compute_network_performance_precise(self):
        return -self.get_reward_matricial_alt(self.C)

def reduce_to(config, max_configs):
    config_ = np.zeros_like(config)
    assert 1 < max_configs < 10
    n = config.shape[1]
    for nodo in range(n):
        best_n = np.argsort(config[:, nodo])[::-1][:max_configs]
        config_[best_n, nodo] = config[best_n, nodo]
        new_sum = config_[:, nodo].sum()
        config_[:, nodo] /= new_sum

    return config_


def run():
    np.random.seed(7)

    num_nodos = 5  # en lugar de 60 nodos con DC = 1% -> 6 con DC = 0.1%?
    lambdas_ = 1 / np.array([60, 130, 100, 200, 120])
    lengths = np.array([20, 25, 28, 15, 17])
    priorities = np.array([1, 3, 0.5, 10, 0])
    snrs = np.array([6.45764192139738, 0.8564039408866996, 7.693981481481482, -8.977067669172932, -0.5])
    forced_nodes = [4]
    forced_configs = np.zeros((49, lambdas_.shape[0]))
    forced_configs[19, 4] = 1
    model_paths = ['raspi/model_cambridge_1.p', 'raspi/model_cambridge_2.p', 'raspi/model_cambridge_3.p',
                   'raspi/model_cambridge_4.p', 'raspi/model_cambridge_4.p']

    # num_nodos = 4  # en lugar de 60 nodos con DC = 1% -> 6 con DC = 0.1%?
    # lambdas_ = 1 / np.array([60, 130, 100, 200])
    # lengths = np.array([20, 25, 28, 15])
    # priorities = np.array([1, 3, 0.5, 10])
    # snrs = np.array([6.45764192139738, 0.8564039408866996, 7.693981481481482, -8.977067669172932])
    # forced_nodes = []
    # forced_configs = np.zeros((49, lambdas_.shape[0]))
    # model_paths = ['raspi/model_cambridge_1.p', 'raspi/model_cambridge_2.p', 'raspi/model_cambridge_3.p',
    #                'raspi/model_cambridge_4.p']

    acceptable_tx_powers = np.ones((4, num_nodos))
    # acceptable_tx_powers[3, 1] = 0
    # acceptable_tx_powers[3, (1, 2, 4)] = 0
    # acceptable_tx_powers[2, 1] = 0

    C_opt = np.zeros((49, lambdas_.shape[0]))
    lora = LoRaWorld(lambdas_, lengths, priorities, snrs, C_opt, 1, model_paths, acceptable_tx_powers, forced_nodes=forced_nodes, forced_configs=forced_configs)

    t1 = time.time()
    # lora.find_optimal_c_13()
    lora.find_optimal_c()
    print('Elapsed computing with Scipy {}'.format(time.time() - t1))
    lora.C = lora.C_opt

    print(lora.compute_network_performance())
    # print(lora.compute_network_performance_precise())


    new_config = reduce_to(lora.C_opt, 5)
    lora.C = new_config
    lora.C_opt = new_config
    print(lora.compute_network_performance())


    # pickle.dump(lora.C_opt, open('opt_config.p', 'wb'))
    # mu = np.zeros((13, lora.N))
    # mu[0, :] = lora.C_opt[0, :]
    # mu[1:, :] = lora.C[1 + 12 * 3: 1 + 12 * 4, :]
    # mu = mu.flatten()


    print(lora.C_opt)

    exit(-1)

if __name__ == '__main__':
    run()