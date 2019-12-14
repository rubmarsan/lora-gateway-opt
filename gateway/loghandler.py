import base64
import re
from math import ceil
import warnings
import dateutil.parser
import numpy as np
from scipy.interpolate import interp1d


class LogHandler():
    """
    Simple class in charge of writing messages to a log file
    """
    def __init__(self, out_path, append=False):
        if append:
            self.file = open(out_path, 'a+')
        else:
            self.file = open(out_path, 'w+')

    def append_msg(self, msg):
        try:
            self.file.write(msg + '\n')
            self.file.flush()
        except ValueError:
            print("could not write")

    def close(self):
        self.file.close()


class LogReader():
    """
    Simple class to parse a log file
    """

    def compute_over_the_air_time(self, payload_length, sf, cr):
        """
        Computes the Time on Air (ToA) of a LoRa packet based on its length and transmission parameters
        :param payload_length: Payload length in bytes
        :param sf: Spreading factor (from 7 to 12)
        :param cr: Coding Ratio (from 5 to 7)
        :return: The ToA in seconds of the packet
        """

        BW = 125e3
        explicit_header = 1
        preamble_symbols = 8
        header_length = 0
        payload_length += 13

        if sf == 0 and cr == 0:
            return 0

        assert 7 <= sf <= 12
        assert 5 <= cr <= 7
        de = 1 if sf >= 11 else 0
        # http://forum.thethingsnetwork.org/t/spreadsheet-for-lora-airtime-calculation/1190/15
        t_sym = pow(2, sf) / BW * 1000  # symbol time in ms
        t_preamble = (preamble_symbols + 4.25) * t_sym  # over the air time of the preamble
        payload_symbol_number = 8 + max([(ceil(
            (8 * (payload_length + header_length) - 4 * sf + 28 + 16 - 20 * (1 - explicit_header)) / (
                4 * (sf - 2 * de))) * cr), 0])  # number of symbols of the payload

        t_payload = payload_symbol_number * t_sym  # payload time in ms
        t_packet = t_preamble + t_payload

        return t_packet / 1000  # expressed in seconds

    def compute_power_consumption(self):
        """
        Computes the total power consumed in transmitting the messages stored in self.MSGs
        :return: Cummulatie power consumed (in Joules) and the instant at which those consumptions were registered
        """
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

        consumed = []
        base_time = dateutil.parser.parse(self.MSGs[0]['time']).timestamp()
        times = []
        for msg in self.MSGs:
            if msg['recv']['TXP'] is None:
                continue
            ota = self.compute_over_the_air_time(4, msg['SF'], msg['CR'])
            consumed.append(ota * current_consumption[msg['recv']['TXP']] * 3.3 / 1000)  # in Joules
            elapsed = dateutil.parser.parse(msg['time']).timestamp() - base_time
            times.append(elapsed)

        return np.cumsum(consumed), np.array(times)

    def compute_uncertainty(self):
        """
        Compute the certainty over the PRR model at each time. Certainty is the difference between the upper and the
        lower bound.
        :return: Array of certainties and an Array of the times at which such certainties were registered
        """
        base_time = dateutil.parser.parse(self.MSGs[0]['time']).timestamp()

        uncertainties = list()
        times = list()

        for msg in self.certainty:
            uncertainty = msg['uncertainty']
            elapsed = dateutil.parser.parse(msg['time']).timestamp() - base_time
            uncertainties.append(uncertainty)
            times.append(elapsed)

        uncertainties.insert(0, 1)
        times.insert(0, 0)

        return np.array(uncertainties), np.array(times)

    def proyect_a_into_b(self, a_t, a, b_t, b):
        """
        For two real-valued functions (a and b) that are evaluated in different moments (a_t and b_t respectively),
        ''proyect_a_into_b'' proyects b into a such as we can evaluate how b correlates with a

        :param a_t: The instants at which the function ''a'' was evaluated
        :param a: The values of function ''a'' taken at ''a_t'' times
        :param b_t: The instants at which the function ''b'' was evaluated
        :param b: The values of function ''b'' taken at ''b_t'' times
        :return: a, the proyection of b into a
        """
        new_x = list(a)
        f = interp1d(b_t, b)
        new_y = [f(t) for t in a_t]

        # new_y = list()
        # for t in a_t:
        #     val_y = f(t)
        #     new_y.append(val_y)


        return new_x, new_y

    def dr_to_sf(self, data_rate):
        """
        Converts from data_rate (0-5) to Spreading Factor (7-12)
        :param data_rate: The data rate at which the communication is taking place
        :return: The corresponding Spreading Factor as an integer from 7 to 12
        """
        return 7 + (5 - data_rate)

    def parse_line(self, line):
        """
        Parses a line of the log file
        :param line: The text line
        :return: A dictionary that contains the important parameters of the packet
        """
        rssi_reg = re.compile("rssi=([-.0-9]+)")
        snr_reg = re.compile("snr=([-.0-9]+)")
        datarate_reg = re.compile("data_rate=\'SF([\d]+)BW[\d]+\'")
        bw_reg = re.compile("data_rate=\'SF[\d]+BW([\d]+)\'")
        codingrate_reg = re.compile("coding_rate=\'([0-9\/]+)\'")
        payload_reg = re.compile("payload_raw=\'((?:[A-Za-z0-9+\/]{4})*(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=)?)\'")
        time_reg = re.compile("time='([^\']*)'")
        # paq_bytes = base64.b64decode(msg.payload_raw)

        rssi = int(rssi_reg.findall(line)[0])
        snr = float(snr_reg.findall(line)[0])
        datarate = int(datarate_reg.findall(line)[0])
        bw = int(bw_reg.findall(line)[0])
        codingrate = int(codingrate_reg.findall(line)[0][-1])

        if codingrate == 8:
            codingrate = 7

        payload_bytes = base64.b64decode(payload_reg.findall(line)[0])
        time = time_reg.findall(line)[0]

        if len(payload_bytes) != 5:
            tx_power_r = counter_r = req_ack = None
            CR = SF = None
        else:
            data_rate_r, coding_rate_r, tx_power_r, counter_r, req_ack = [int(v) for v in payload_bytes]
            SF = self.dr_to_sf(data_rate_r)
            CR = coding_rate_r + 4

        ret = {
            'rssi': rssi,
            'snr': snr,
            'SF': datarate,
            'BW': bw,
            'CR': codingrate,
            'time': time,
            'recv': {'SF': SF,
                     'CR': CR,
                     'TXP': tx_power_r,
                     'counter': counter_r,
                     'req_ack': req_ack
                     }
        }

        return ret

    def parse_certainty(self, line):
        """
        Parses an "uncertainty" text line with the following format (example):
        Current uncertainty is: 0.83838383. Time = '2018-07-30T12:34:09Z'
        :param line: The text line
        :return: A dictionary with keys uncertainty and time
        """
        time_reg = re.compile("Time = '([^\']*)'")
        uncertainty_reg = re.compile("is: ([\d].[\d]+)")

        time = time_reg.findall(line)[0]
        uncertainty = uncertainty_reg.findall(line)[0]

        return {'uncertainty': float(uncertainty), 'time': time}

    def parse_prr(self, line, time, correct=False):
        """
        Parses a "PRR" line with the following format:
        PRR SO FAR: 1.0 (7/7)
        :param line: The text line
        :param time: The time at which such a line was received
        :param correct: Whether to correct errorrs
        :return: A dictionary with PRR, time, received packets, total sent packets
        """
        prr_reg = re.compile("PRR SO FAR: [\d.]+ \(([\d]+)\/([\d]+)\)")
        pos = int(prr_reg.findall(line)[0][0])
        tot = int(prr_reg.findall(line)[0][1])

        warnings.warn("Hardcoded corrections!")
        if "adr" not in self.file.name:
            if "cambridge_1" in self.file.name and tot >= 89:
                tot -= 1

            if "cambridge_2" in self.file.name and tot >= 65:
                tot -= 1

            if "cambridge_3" in self.file.name and tot >= 56:
                tot -= 1

        prr = pos / tot

        timestamp = dateutil.parser.parse(time).timestamp()

        return {'PRR': prr, 'time': timestamp, 'pos': pos, 'tot': tot}

    def __init__(self, in_path):
        """
        Initializes the log parser object.
        :param in_path: The path of the log file
        """
        self.file = open(in_path, 'r')
        # lines = self.file.readlines()[0]
        # msgs_raw = lines.split('MSG(dev_id=')[1:]
        # msgs_raw = ['MSG(dev_id=' + msg for msg in msgs_raw]

        lines = self.file.readlines()

        self.MSGs = list()
        self.certainty = list()
        self.PRRs = list()

        for line in lines:
            if line.startswith('MSG('):
                msg = self.parse_line(line)
                if msg is not None:
                    self.MSGs.append(msg)

            if line.startswith('Current uncertainty is'):
                certainty = self.parse_certainty(line)
                self.certainty.append(certainty)

            if line.startswith('PRR SO FAR'):
                prr = self.parse_prr(line, time=self.MSGs[-1]['time'], correct=False)
                self.PRRs.append(prr)

        print('Messages parsed')

    def get_snr(self, tx_power):
        """
        Get the SNR for a given transmission power
        :param tx_power: The Transmission power employed
        :return: The average SNR for such a tx_power
        """
        assert tx_power in range(2, 15)
        selection = [v['snr'] for v in self.MSGs if v['recv']['TXP'] == tx_power]
        return np.mean(selection), np.std(selection)

    def close(self):
        """
        Closses the log file
        :return: None
        """
        self.file.close()


def moving_average(a, n=3):
    """
    A simple moving average
    :param a: The list containing the values
    :param n: The window size
    :return: The list ''a'' averaged using a sliding window
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n  # L - n + 1


def re_think_prr(prrs):
    """
    Instead of considering the PRR of the TOTAL sent packets, use a sliding window to compute the PRR of the last N
    packets
    :param prrs: The PRR list
    :return: The PRR with the sliding window technique applied
    """
    l = [True] * 9
    pre = 1
    for prr in prrs:
        pos = prr['pos']
        tot = prr['tot']
        assert np.isclose(prr['PRR'], pos / tot, 1e-3)
        if prr['PRR'] < pre:
            l.append(False)
        else:
            l.append(True)

        pre = prr['PRR']

    smoothed = moving_average(l, 10)
    return [{'PRR':s, 'time': p['time']} for (s, p) in zip(smoothed, prrs)]


def average_PRRs(prr_list):
    """
    Given a list of PRR values (a list of N dicts, where N is the number of nodes) average their values. For instance,
    we have two nodes in the network so:
    prr_list = [    {1: 0.9, 2: 0.8, 3: 0.9},
                    {1: 1, 2: 0.8, 3: 0.8}
                ]
    this function should average their values so -> {1: 0.95, 2: 0.8, 3: 0.85}. However, it returns keys and values
    of the dictionary in two different objects. Furthermore, it extrapolates values so if PRRs for the 1st node are
    evaluted in different times than 2nd node, it can all be averaged.
    :param prr_list: The list of dicts with the PRRs of each node.
    :return: The times at which the PRRs are evaluted, the average of such PRRs
    """
    assert (len(prr_list) > 1)

    fs = []

    lowest_time = float("inf")
    highest_time = float("-inf")

    for prrs in prr_list:
        times = [v['time'] for v in prrs]
        values = [v['PRR'] for v in prrs]

        lowest_time = min(lowest_time, min(times))
        highest_time = max(highest_time, max(times))

        fs.append(interp1d(times, values, fill_value=(values[0], values[-1]), bounds_error=False))

    time_scale = np.linspace(lowest_time, highest_time, num=1000)
    avg_PRRs = []
    for t in time_scale:
        avg_PRRs.append(np.mean([f(t) for f in fs]))

    return time_scale - lowest_time, avg_PRRs


def run():
    from matplotlib import pyplot as plt

    # # # # Para ver el PRR con (ADR vs Proposed Method)run 4,.5
    lr_1 = LogReader(in_path="Result run 4.5/log_normal_cambridge_1.csv")
    lr_2 = LogReader(in_path="Result run 4.5/log_normal_cambridge_2.csv")
    lr_3 = LogReader(in_path="Result run 4.5/log_normal_cambridge_3.csv")
    lr_4 = LogReader(in_path="Result run 4.5/log_normal_cambridge_4.csv")

    new_prrs = [re_think_prr(lr_1.PRRs), re_think_prr(lr_2.PRRs), re_think_prr(lr_3.PRRs), re_think_prr(lr_4.PRRs)]
    # new_prrs = [(lr_1.PRRs), (lr_2.PRRs), (lr_3.PRRs), (lr_4.PRRs)]
    t, p = average_PRRs(new_prrs)

    print('ok')
    fig, ax = plt.subplots()


    # t = np.append(t, 7900)
    # p = np.append(p, 1)
    ax.plot(t, p, 'b', linewidth=2)
    ax.set_ylabel('PRR (%)', fontsize=15)
    ax.set_xlabel('Time (minutes)', fontsize=15)
    ax.set_ylim([0.82, 1.05])
    ax.grid(True)
    ax.set_title('PRR vs time', fontsize=17)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='b', lw=3, label='PRR vs Time'),
                       Line2D([0], [0], color='g', lw=3, linestyle="dashed", label='Increase of interference')]

    ax.legend(handles=legend_elements, loc='lower left', fontsize='x-large', framealpha=0.5)
    ax.axvline(x=3350, linestyle="dashed", c='green', lw=2)   # for mio
    # ax.axvline(x=3730, linestyle="dashed", c='green', lw=2) # for adr
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=13)

    # old_pos_ticks = ax.get_xticks()
    new_pos_ticks = np.linspace(0, 135 * 60, 11)
    new_label_ticks = ['{:0d}'.format(int(v / 60)) for v in new_pos_ticks]
    ax.set_xticklabels(new_label_ticks)

    plt.savefig('experiment_mio.png', dpi=100)
    plt.savefig('experiment_mio.eps', dpi=100)
    plt.show()
    exit()


    # # # # Para ver como evoluciona el RSSI con el tiempo
    # lr = LogReader(in_path="log_5_all-updated.csv")
    # snrs = [v['rssi'] for v in lr.MSGs]
    # times = np.array([dateutil.parser.parse(v['time']).timestamp() for v in lr.MSGs])
    #
    # smooth_factor = 10
    # snrs_smoothed = moving_average(snrs, smooth_factor)
    # times_smoothed = times[smooth_factor-1:]
    #
    # plt.hist(snrs)
    # plt.show()
    # plt.plot(times_smoothed - times_smoothed[0], snrs_smoothed - np.mean(snrs_smoothed))
    # plt.show()
    # exit()

    # # # # Para ver la evolución del modelo vs time
    lr = LogReader(in_path="raspi2/log_cambridge_9.csv")
    pc, pct = lr.compute_power_consumption()
    pc_sorted_idx = np.argsort(pc)
    pc = pc[pc_sorted_idx]
    un_1, unt_1 = lr.compute_uncertainty()
    un_1_sorted_idx = np.argsort(un_1)[::-1]
    un_1 = un_1[un_1_sorted_idx]
    # unt = unt[un_sorted_idx]

    plt.figure()
    plt.plot(unt_1, un_1 * 100, linewidth=2)
    plt.title('Uncertainty in PRR vs time', fontsize=14)
    plt.xlabel('Time (minutes)', fontsize=13)
    plt.ylabel('Uncertainty in PRR (%)', fontsize=13)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    plt.grid(True)
    old_pos_ticks, _ = plt.xticks()
    new_pos_ticks = np.arange(0, old_pos_ticks[-1] + 60, 60*4)
    new_label_ticks = ['{:0d}'.format(int(1.25*v/60)) for v in new_pos_ticks]

    plt.xticks(new_pos_ticks, new_label_ticks)
    plt.savefig('uncertainty_vs_time.png', dpi=100)
    plt.savefig('uncertainty_vs_time.eps', dpi=100)

    # plt.figure()
    # plt.plot(pct, pc)
    # plt.title('Power consumption vs time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Power Consumpton (Joules)')
    # plt.grid(True)
    #
    x, y = lr.proyect_a_into_b(pct, pc, unt_1, un_1)
    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.title('Uncertainty vs power consumption', fontsize=14)
    plt.xlabel('Power Consumption (Joules)', fontsize=13)
    plt.ylabel('Uncertainty in PRR (%)', fontsize=13)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=13)
    plt.tick_params(axis='both', which='minor', labelsize=13)
    plt.savefig('uncertainty_vs_pc.png', dpi=100)

    plt.show()

    print("2", lr.get_snr(14))


if __name__ == '__main__':
    run()
