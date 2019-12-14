import base64
import datetime
import pickle
from collections import namedtuple
from math import ceil
from threading import Thread
import os
from functools import partial
import sys
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np

from worker_sweep import Worker


class Model():
    """
    LNSPL Model for estimating path losses
    """

    def __init__(self, BPL, n, sigma, mean_noise, sigma_noise):
        """
        LNSPL Model for estimating path losses
        :param BPL: Base path losses. Losses (in dBm) at the reference distance (1km)
        :param n: The path-loss exponent
        :param sigma: The standard deviation of the model
        :param mean_noise: Mean of the noise floor
        :param sigma_noise: Standard deviation of the noise floor
        """
        self.BPL = BPL
        self.n = n
        self.sigma = sigma
        self.noise = type('noise', (object,), {})()  # lambda: None    # chapu
        self.noise.mean = mean_noise
        self.noise.sigma = sigma_noise

        self.alphas = np.array(
            [0, -30.2580, -77.1002, -244.6424, -725.9556, -2109.8064, -4452.3653, -105.1966, -289.8133, -1114.3312,
             -4285.4440, -20771.6945, -98658.1166])
        self.betas = np.array(
            [0, 0.2857, 0.2993, 0.3223, 0.3340, 0.3407, 0.3317, 0.3746, 0.3756, 0.3969, 0.4116, 0.4332, 0.4485])

    def compute_prr_snr_rssi(self, packet_length, sf, cr, distance, txp):
        """
        Compute the PRR, SNR, and RSSI of the communication
        :param packet_length: Packet length (in bytes) of the transmission
        :param sf: Spreading Factor index (from 7 to 12)
        :param cr: Coding Ratio index (either 5 or 7)
        :param distance: Distance in Km between the node and the Gateway
        :param txp: Transmission power in dBm of the communication
        :return: PRR, SNR, and RSSI
        """

        # packet_length in bytes
        # distance in Km
        # BPL is measured at 1 Km
        # txp in dBm
        # sf in [7, 12]
        # cr in [5, 8]

        if sf == 0 and cr == 0:
            return 0

        snr, rssi = self.__get_snr_rssi__(distance, txp)

        # assert cr in (5, 7), "CR not implemented yet"
        # assert 7 <= sf <= 12, "Invalid SF"
        if cr not in (5, 7):
            if cr == 6:
                cr = 5
            elif cr == 8:
                cr = 7
            else:
                raise ValueError("Invalid CR")

        if sf not in range(7, 13):
            print("Invalid SF")
            raise ValueError("Invalid SF")

        action = (sf - 6) + [0, 6][cr == 7]
        alfa = self.alphas[action]
        beta = self.betas[action]
        ber = np.power(10, alfa * np.exp(beta * snr))
        return np.power(1 - ber, packet_length * 8), snr, rssi

    def __get_snr_rssi__(self, distance, txp):
        """
        Internal method to compute the SNR and RSSI based on distance and TXP
        :param distance: Distance in Km between node and gateway
        :param txp: Transmission power in dbm of the communciation
        :return: SNR and Reception Power (RSSI)
        """
        losses = self.BPL + 10 * self.n * np.log10(distance / 1) + np.random.normal(scale=self.sigma)
        gain = 2.15
        if txp + gain > 16:
            txp -= 1
        rxp = txp + gain - losses
        noise = np.random.normal(loc=self.noise.mean, scale=self.noise.sigma)
        snr = rxp - noise
        return snr, rxp


class TabularModel():
    """
    Returns the PRR for a particular configuration when we have the model (obtained by worker_sweep.py)
    """

    def __init__(self, path_to_model):
        self.model = pickle.load(open(path_to_model, 'rb'))

    def compute_prr_snr_rssi(self, packet_length, sf, cr, txp, distance):
        # packet_length in bytes
        # distance in Km
        # BPL is measured at 1 Km
        # txp in dBm
        # sf in [7, 12]
        # cr in [5, 8]
        return self.model[sf, cr - 4, txp] ** (packet_length / 18), 0, 0


def MSG(payload_bytes, counter, datetime, data_rate, coding_date, rssi, snr):
    """
    Re-implementation of the MSG class of TTN MQTT hidden and buried somewhere in their code
    :param payload_bytes: Payload bytes (bytes-array)
    :param counter: Counter of the packet (up to 2**16-1)
    :param datetime: Timestamp
    :param data_rate: Data Rate index (from 7 to 12)
    :param coding_date: Coding Ratio index (from 5 to 8)
    :param rssi: RSSI of the transmission
    :param snr: SNR of the transmission
    :return:
    """
    payload_fields = {'error': 'empty'}
    gateways = {'gtw_id': 'gateway_under_test', 'gtw_trusted': True, 'timestamp': datetime.timestamp(),
                'time': datetime.isoformat(), 'channel': 0, 'rssi': rssi, 'snr': snr, 'rf_chain': 1, 'latitude': 0,
                'longitude': 0, 'location_source': 'registry'}
    metadata = {'time': datetime.isoformat(), 'frequency': '868.1', 'modulation': 'LORA',
                'data_rate': 'SF' + str(data_rate) + 'BW125', 'airtime': 0, 'coding_rate': '4/' + str(coding_date),
                'gateways': [convert(gateways)]}

    payload_raw = base64.b64encode(payload_bytes).decode('utf-8')
    msg_dict = {'app_id': 'app-prueba-2', 'dev_id': 'cambridge_1', 'hardware_serial': 'XXXXXXXX', 'port': 2,
                'counter': counter, 'payload_raw': payload_raw, 'payload_fields': convert(payload_fields),
                'metadata': convert(metadata)}

    msg = convert(msg_dict)
    return msg


def convert(dictionary):
    """
    Converts a dictionary to a named tuple
    :param dictionary: the dictionary to be converted
    :return: the named tuple
    """
    return namedtuple('MSG', dictionary.keys())(**dictionary)


def compute_over_the_air_time(payload_length, sf, cr):
    """
    Compute the time over-the-air (ToA) of the transmission
    :param payload_length: Length of the packet in bytes (also considering headers!)
    :param sf: Spreading Factor index (from 7 to 12)
    :param cr: Coding Ratio Index (from 5 to 8)
    :return: ToA in seconds
    """
    # payload_length in bytes
    BW = 125e3
    explicit_header = 1
    preamble_symbols = 8
    header_length = 0

    if sf == 0 and cr == 0:
        return 0

    assert 7 <= sf <= 12
    assert 5 <= cr <= 8
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


class FakeNode():
    """
    This is an implementation of LoPy behaviour in python. It aims to replicate LoPy code behaviour. Therefore, most
    of the code is the same as the LoPy code and hence, it is not commented.
    """

    def __init__(self, worker, dev_id, pathloss_model, distance_to_gw):
        """
        This is an implementation of LoPy behaviour in python. It aims to replicate LoPy code behaviour. Therefore, most
        of the code is the same as the LoPy code and hence, it is not commented.
        :param worker: Instance of the Worker class (either Worker_normal or Worker_sweep)
        :param dev_id: Id of the device
        :param pathloss_model: The pathloss model instance (either Model or TabularModel)
        :param distance_to_gw: Distance in Km to the gateway
        """

        self.worker = worker
        self.dev_id = dev_id
        self.pathloss_model = pathloss_model
        self.dist = distance_to_gw
        self.order = None
        self.tx_power = None
        self.cr = None
        self.dr = None
        self.counter = None
        self.keep_alive_counter = None
        self.hash = None
        self.stop_thread = False
        self.radio_param = namedtuple('LoRa', ['data_rate', 'coding_rate', 'tx_power', 'counter'])
        self.radio_param.counter = 0
        self.packets_sent = list()
        self.certainties = list()
        self.done = False
        self.beeps = 0
        # self.t = datetime.datetime.now().timestamp()

    def run(self):
        # This tries to simulate the infinite loop of the LoPy mote

        while not self.done:
            self.p = Thread(target=self.boot)
            self.p.daemon = True
            self.p.start()
            self.p.join()

        print("Finished, closing up")
        # self.boot()

    def reboot(self):
        print("Rebooting")
        self.stop_thread = True
        # self.p.join()
        # print("Thread terminated, lunching another one")
        # self.p = Thread(target=self.boot)
        # self.p.daemon = True
        # self.p.start()

    def send(self, dev_id, pay, port=1, conf=False, sched="replace"):
        """
        External method for making this node receive info
        :param dev_id: device ID
        :param pay: payload in a base64 string
        :param port: TTN port
        :param conf: Confirmation required (ACK)
        :param sched: scheduling
        :return: None
        """

        paq_bytes = base64.b64decode(pay)
        pkt_length = 14 + len(paq_bytes)
        sf_tx = self.dr_to_sf(self.radio_param.data_rate)
        cr_tx = 4 + self.radio_param.coding_rate
        txp_tx = self.radio_param.tx_power
        prr, snr, rssi = self.pathloss_model.compute_prr_snr_rssi(packet_length=pkt_length, sf=sf_tx, cr=cr_tx,
                                                                  distance=self.dist, txp=txp_tx)

        if np.random.rand() < prr:
            self.__receive__(dev_id, pay, port=port, conf=conf, sched=sched)
        else:
            print("Reception lost")

    def set_tx_power(self, tx_power):
        self.tx_power = tx_power
        self.radio_param.tx_power = tx_power

    def set_coding_rate(self, coding_rate):
        self.cr = coding_rate
        self.radio_param.coding_rate = coding_rate

    def set_data_rate(self, data_rate):
        self.dr = data_rate
        self.radio_param.data_rate = data_rate

    @staticmethod
    def dr_to_sf(data_rate):
        return 7 + (5 - data_rate)

    @staticmethod
    def sf_to_dr(sf):
        return 5 - (sf - 7)

    def __send__(self, bytes_raw):
        """
        Internal method for this method to send information
        :param bytes_raw: Byte-array of the payload
        :return: None
        """
        dt = datetime.datetime.now()

        pkt_length = 18
        sf_tx = self.dr_to_sf(self.radio_param.data_rate)
        cr_tx = 4 + self.radio_param.coding_rate
        txp_tx = self.radio_param.tx_power
        # Determines the PRR, SNR, RSSI
        prr, snr, rssi = self.pathloss_model.compute_prr_snr_rssi(packet_length=pkt_length, sf=sf_tx, cr=cr_tx,
                                                                  distance=self.dist, txp=txp_tx)

        # tosses a coin to finaly decide whether this packet got lost or not
        if np.random.rand() < prr:
            msg = MSG(bytes_raw, counter=self.radio_param.counter, datetime=dt,
                      data_rate=self.dr_to_sf(self.radio_param.data_rate), coding_date=self.radio_param.coding_rate + 4,
                      rssi=int(round(rssi)), snr=snr)
            self.worker.uplink_callback(msg, self)
        else:
            print("Sweep lost")

        # regardles of this packet was received or not successfully, it must be accounted for the power consumption
        self.packets_sent.append({'SF': self.dr_to_sf(self.radio_param.data_rate),
                                  'CR': self.radio_param.coding_rate + 4,
                                  'TXP': self.radio_param.tx_power,
                                  'length': 18})
        self.certainties.append(self.worker.compute_mean_uncertainty())

        self.radio_param.counter += 1

    def __receive__(self, dev_id, pay, port=1, conf=False, sched="replace"):
        """
        Internal method through which this node receives info
        :param dev_id: device ID
        :param pay: payload in a base64 string
        :param port: TTN port
        :param conf: Confirmation required (ACK)
        :param sched: scheduling
        :return: None
        """
        recv_bytes = base64.b64decode(pay)
        if len(recv_bytes) == 3:
            # this kind of packets is received when we are sweeping the channel to construct the PRR
            self.keep_alive_counter = 0
            data_rate, coding_rate, tx_power = recv_bytes
            print("Received response from beacon")
            print(self.dr_to_sf(data_rate), coding_rate, tx_power)

            if data_rate == 255 and coding_rate == 0 and tx_power == 255:
                print("received signal to stop! I am done!")
                self.stop_thread = True
                self.done = True
                return

            if 2 <= tx_power <= 14:
                if tx_power != self.tx_power:
                    self.set_tx_power(tx_power)
                    print("new tx_power", tx_power)
                    self.counter = -1
                    self.radio_param.counter = 0
                else:
                    print("Same tx power")
            else:
                print("tx_power exceeded limtis", tx_power)

            if 0 <= data_rate <= 5:
                if data_rate != self.dr:
                    self.set_data_rate(data_rate)
                    print("new data_rate", data_rate)
                    self.counter = -1
                    self.radio_param.counter = 0
                else:
                    print("Same data rate")
            else:
                print("data_rate exceeded limits", data_rate)

            if 1 <= coding_rate <= 4:
                if coding_rate != self.cr:
                    self.set_coding_rate(coding_rate)
                    print("new coding_rate", coding_rate)
                    self.counter = -1
                    self.radio_param.counter = 0
                else:
                    print("Same coding rate")
            else:
                print("coding_rate exceeded limits", coding_rate)
        elif len(recv_bytes) >= 4:
            # new order received from the GW
            if recv_bytes[0] == 1:
                # order is None
                self.order = None
                self.hash = None
                self.reboot()
            elif recv_bytes[0] == 2:
                # order is sweep
                data_rate, coding_rate, tx_power = recv_bytes[1:4]
                if self.dr == data_rate and self.cr == coding_rate and self.tx_power == tx_power and self.order == 2:
                    print("Duplicate sweep order")
                    return

                self.set_tx_power(tx_power)
                self.set_data_rate(data_rate)
                self.set_coding_rate(coding_rate)
                self.order = 2

                assert 0 <= data_rate <= 5
                assert 1 <= coding_rate <= 4
                assert 2 <= tx_power <= 14

                print("Received order SWEEP from GW.")
                print("\tData rate: {}.\n\tCoding rate: {}.\n\tTX power: {}.".format(data_rate, coding_rate, tx_power))

                self.reboot()
            elif recv_bytes[0] == 3:
                # order is normal transmission (not implemented in this desktop-python version)
                raise Exception("Not implemented yet")
            else:
                raise Exception("Unrecognised order")

    def boot(self):
        print('Node Initialized')
        self.counter = -1
        self.keep_alive_counter = 0
        self.radio_param.counter = 0
        while True:
            print('Restarted')

            if self.stop_thread:
                # used when we want to turn off the device
                print("Called to be killed :[")
                self.stop_thread = False
                return

            # self.t += 3 # node takes approximately 3 seconds to boot
            if self.order is None or self.order == 0:
                # order is zero, keep beeping to receive a new order from the Gateway
                self.set_tx_power(14)
                self.set_coding_rate(1)
                self.set_data_rate(0)

                while self.order is None or self.order == 0:
                    if self.stop_thread:
                        print("Called to be killed :[")
                        self.stop_thread = False
                        return

                    print("Beeping to get a new order")
                    self.__send__(bytes([0, 1, 2]))
                    self.beeps += 1
                    print("Ended sending")

                    if self.beeps > 50:
                        print("Beeped for too long")
                        self.stop_thread = False
                        self.done = True
                        return
                        # if self.order is None or self.order == 0:
                        #     self.t += 30
            elif self.order == 2:
                print("Stored order is param sweep")
                print("\tData rate: {}\n\tCoding rate: {}\n\tTX power: {}".format(self.dr, self.cr, self.tx_power))
                self.set_tx_power(self.tx_power)
                self.set_coding_rate(self.cr)
                self.set_data_rate(self.dr)

                while True:
                    if self.stop_thread:
                        print("Called to be killed :[")
                        self.stop_thread = False
                        return

                    if self.keep_alive_counter > 20:
                        # did not receive any keep-alive packet from the Gateway in more than 20 transmissions
                        # prepare the bomb!
                        print("About to start bombing...")

                        b_dr = self.dr
                        b_cr = self.cr
                        b_tx_power = self.tx_power

                        if self.dr == 0 and self.cr == 4:
                            self.set_tx_power(14)
                            self.set_coding_rate(3)
                            self.set_data_rate(0)
                        else:
                            self.set_tx_power(14)
                            self.set_coding_rate(4)
                            self.set_data_rate(0)

                        self.dr = b_dr
                        self.cr = b_cr
                        self.tx_power = b_tx_power

                        for _ in range(100):
                            if self.stop_thread:
                                print("Called to be killed :[")
                                return

                            self.counter += 1
                            self.keep_alive_counter += 1

                            paq_bytes = bytes([self.dr, self.cr, self.tx_power, self.counter & 0xff, 1])

                            if self.keep_alive_counter <= 20:
                                break
                            else:
                                print("Still no response...")
                                print("Sending this bomb once again")

                            self.__send__(paq_bytes)

                        # restoring self.radio_param
                        self.set_tx_power(self.tx_power)
                        self.set_coding_rate(self.cr)
                        self.set_data_rate(self.dr)

                        if self.keep_alive_counter > 20:
                            # if after 100 attempts, did not receive anything, we better die in peace
                            print("Could not contact with Gateway after 100 attempts. Breaking.")
                            break
                        else:
                            print("Restored from bombing state")
                            self.keep_alive_counter = 0

                    self.counter += 1
                    self.keep_alive_counter += 1
                    paq_bytes = bytes([self.dr, self.cr, self.tx_power, self.counter, self.keep_alive_counter > 10])
                    print("sending sweep beacon with", self.dr_to_sf(self.dr), self.cr, self.tx_power)
                    self.__send__(paq_bytes)

                print("End sweeping params, clearing order and restarting")
                self.order = 0

            if self.stop_thread:
                print("Called to be killed :[")
                self.stop_thread = False
                return
            else:
                print("Ended, reseting")
                # reseting not needed as we are inside a "While True" loop

    def compute_power_consumption(self):
        """
        Method to compute power consumption for all sent packets
        The specific mA drawn from each TXP value are as per the accompanying article.
        Node is assumed to be powered with 3.3V
        :return: power consumed (in Joules) for transmitting all packets
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
        for pkt in self.packets_sent:
            ota = compute_over_the_air_time(pkt['length'], pkt['SF'], pkt['CR'])
            consumed.append(ota * current_consumption[pkt['TXP']] * 3.3 / 1000)  # in Joules

        return np.cumsum(consumed)


def run(distance, runs, min_prr_testable=0.3):
    """
    For the given distance and number of runs, compute the power required to obtain the PRR model
    :param distance: distance from node to gateway
    :param runs: number of iterations
    :param min_prr_testable: If PRR is believed to be below this threshold it won't be tested
    :return: The power required to obtain the PRR model for such a distance evaluated ``runs`` times
    """
    results = []
    for run in range(runs):
        np.random.seed(run)
        sfs = [7, 8, 9, 10, 11, 12]
        crs = [1, 3]
        tx_powers = [2, 6, 10, 14]
        certainty = 0.2
        m = Model(BPL=128.95, n=2.32, sigma=7.8, mean_noise=-108, sigma_noise=1.3)
        # m = TabularModel("raspi2/model_cambridge_9.p")
        worker = Worker('cambridge_1', sfs, crs, tx_powers, certainty, gw_id='gateway_under_test',
                        min_prr_testable=min_prr_testable, brute_force=min_prr_testable <= 0)
        node = FakeNode(worker, 'cambridge_1', m, distance_to_gw=distance)
        node.run()

        ce = node.compute_power_consumption()
        # print("Total consumed energy at {} kms:\n\t{}\nAverage uncertainty:\n\t{}".format(distance, ce[-1], worker.compute_mean_uncertainty()))
        results.append(ce[-1])

    print("Distance {} done".format(distance))
    return results


if __name__ == '__main__':
    if not os.path.exists('result_consumption_crs.p'):
        print("Going with the bounding technique")
        num_distances = 20  # 20 samples from 0.1Km to 15Km
        num_points = 30  # average with 30 iterations

        distances = np.linspace(0.1, 15, num_distances, endpoint=True)
        run_averaged = partial(run, runs=num_points, min_prr_testable=0.3)

        back_sys_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')

        with ThreadPoolExecutor() as executor:
            result = executor.map(run_averaged, distances)

        result = list(result)
        result_dict = {distances[v]: result[v] for v in range(num_distances)}
        sys.stdout = back_sys_stdout

        print('Average power consumption per distance:')
        for d in result_dict.keys():
            print("Distance {}, average power consumption {}".format(d, np.mean(result_dict[d])))

        pickle.dump(result_dict, open('result_consumption_crs.p', 'wb'))

    if not os.path.exists('result_consumption_crs_bf.p'):
        print("Going with the brute force approach")
        num_distances = 20 # 20 samples from 0.1Km to 15Km
        num_points = 30 # average with 30 iterations

        distances = np.linspace(0.1, 15, num_distances, endpoint=True)
        run_averaged = partial(run, runs = num_points, min_prr_testable=0)

        back_sys_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')

        with ThreadPoolExecutor() as executor:
            result = executor.map(run_averaged, distances)

        result = list(result)
        result_dict = {distances[v]: result[v] for v in range(num_distances)}
        sys.stdout = back_sys_stdout

        print('Average power consumption per distance:')
        for d in result_dict.keys():
            print("Distance {}, average power consumption {}".format(d, np.mean(result_dict[d])))

        pickle.dump(result_dict, open('result_consumption_crs_bf.p', 'wb'))

    # # # new python instances can start here
    result_dict = pickle.load(open('result_consumption_crs.p', 'rb'))
    result_dict_bf = pickle.load(open('result_consumption_crs_bf.p', 'rb'))
    plt.plot(list(result_dict_bf.keys()), [np.mean(result_dict_bf[v]) for v in result_dict_bf.keys()], 'k--',
             linewidth=2)
    plt.plot(list(result_dict.keys()), [np.mean(result_dict[v]) for v in result_dict.keys()], 'k-', linewidth=2)
    leg = plt.legend(['Brute-force approach', 'Proposed approach'], fontsize='large')
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.xlabel('Distance (in Km) between node and gateway', fontsize=13)
    plt.ylabel('Energy consumption (in Joules)\n to build the model', fontsize=13)
    plt.title('Energy consumption vs distance', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Energy_consumption_vs_distance_crs.png', dpi=300)
    plt.show()
