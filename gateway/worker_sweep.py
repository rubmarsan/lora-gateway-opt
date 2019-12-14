import base64
import datetime
import pickle
from collections import OrderedDict
from itertools import product

import ncephes
import numpy as np

from loghandler import LogHandler


class Worker():
    """
    Worker sweep. His main duty is to construct the PRR model
    """
    def get_pos_interval(self, m, n, c):
        """
        Computes the positive interval for the Bernoulli distribution according to
        https://arxiv.org/pdf/1105.1486.pdf
        :param m: the positive cases (number of received packets)
        :param n: total cases (number of packets sent)
        :param c: c * 100% = confidence interval
        :return: With c*100% confidence interval, the mean will be smaller than the returned value
        """
        return ncephes.cprob.incbi(m + 1, n - m + 1, 0.5 * (1 + c))

    def get_neg_interval(self, m, n, c):
        """
        Computes the negative interval for the Bernoulli distribution according to
        https://arxiv.org/pdf/1105.1486.pdf
        :param m: the positive cases (number of received packets)
        :param n: total cases (number of packets sent)
        :param c: c * 100% = confidence interval
        :return: With c*100% confidence interval, the mean will be larger than the returned value
        """
        return ncephes.cprob.incbi(m + 1, n - m + 1, 0.5 * (1 - c))

    def compute_mean_uncertainty(self):
        """
        Computes the mean uncertainty in the estimation of the PRR
        The uncertainty is computed as the difference between the positive interval and the negative interval
        :return: The mean uncertainty of the acquired PRR values
        """
        uncertainties = list()

        for key in self.margins.keys():
            # if np.isnan(self.model[key]):
            #     uncertainties.append(1)
            # else:
            #     uncertainties.append(self.margins[key][2] / 2)
            uncertainties.append(self.margins[key][2])

        return np.mean(uncertainties)

    def bound_upper_rest(self, tx_power, sf, cr, prr):
        """
        Applies the "bounding" technique to the acquired model in the "upper" direction.
        See the article for more information on this technique.
        :param tx_power: TX Power of the configuration for which has a solid measure of the PRR
        :param sf: Spreading Factor of the configuration for which has a solid measure of the PRR
        :param cr: Coding Ration of the configuration for which has a solid measure of the PRR
        :param prr: The measured PRR for such a configuration
        :return: None
        """

        # If going with the brute force approach, bounding should not be done
        if self.brute_force:
            return

        for v in self.margins.keys():
            sf_key, cr_key, tx_power_key = v
            if tx_power_key >= tx_power and sf_key == sf and cr_key >= cr and np.isnan(self.model[v]):
                self.margins[v][0] = prr
                self.margins[v][1] = max(self.margins[v][1], prr)
                self.margins[v][2] = self.margins[v][1] - self.margins[v][0]

                assert self.margins[v][2] >= 0

    def bound_lower_rest(self, tx_power, sf, cr, prr):
        """
        Applies the "bounding" technique to the acquired model in the "lower" direction.
        See the article for more information on this technique.
        :param tx_power: TX Power of the configuration for which has a solid measure of the PRR
        :param sf: Spreading Factor of the configuration for which has a solid measure of the PRR
        :param cr: Coding Ration of the configuration for which has a solid measure of the PRR
        :param prr: The measured PRR for such a configuration
        :return: None
        """

        # If going with the brute force approach, bounding should not be done
        if self.brute_force:
            return

        for v in self.margins.keys():
            sf_key, cr_key, tx_power_key = v
            if tx_power_key <= tx_power and sf_key == sf and cr_key <= cr and np.isnan(self.model[v]):
                self.margins[v][0] = min(self.margins[v][0], prr)
                self.margins[v][1] = prr
                self.margins[v][2] = self.margins[v][1] - self.margins[v][0]

                assert self.margins[v][2] >= 0

    def get_greater_margin(self):
        """
        Gets the configuration for which we have the least information about his PRR.
        Information is measured in terms of uncertainty
        :return: A key in the format (SF, CR, TXP)
        """
        idx = None
        # max_margin = 0
        max_margin = self.certainty
        for key, val in self.margins.items():
            if val[2] > max_margin and np.isnan(self.model[key]):
                if val[1] > self.min_prr_testable:
                    max_margin = val[2]
                    idx = key
                else:
                    print("skipping not testable", key)

        return idx

    def update_params(self):
        """
        Considering the received and sent packets, computes the uncertainty over the current-configuration PRR
        Based on that uncertainty, this function decides whether to continue sensing such a configuration or
        move on to the next one (if there is any)
        :return: the new point (SF, CR, TXP) to scan and whether it is a new point or None if we are done
        """
        # formulas taken from https://arxiv.org/pdf/1105.1486.pdf
        changed = False
        m = self.hits.count(True)
        n = len(self.hits)
        x_pos = self.get_pos_interval(m, n, self.confidence)
        x_neg = self.get_neg_interval(m, n, self.confidence)
        assert x_pos >= x_neg
        certainty = abs(x_pos - x_neg)
        print(self.node_id, "m: {}, n: {}, certainty: {}".format(m, n, certainty))
        # e_x = (m + 1) / (n + 2)
        if certainty < self.certainty and len(self.hits) > 10:  # good enough precision
            self.model[(self.g_sf, self.g_cr, self.g_tx_power)] = m / n
            self.margins[(self.g_sf, self.g_cr, self.g_tx_power)] = [x_neg, x_pos, x_pos - x_neg]

            self.bound_upper_rest(self.g_tx_power, self.g_sf, self.g_cr, min(x_neg, m / n))
            self.bound_lower_rest(self.g_tx_power, self.g_sf, self.g_cr, max(x_pos, m / n))

            new_test_point = self.get_greater_margin()

            # partial saving :D
            pickle.dump(self.model, open('model_{}.p'.format(self.node_id), 'wb'))
            pickle.dump(self.margins, open('margins_{}.p'.format(self.node_id), 'wb'))
            self.lh.append_msg(
                "Current uncertainty is: {}. Time = '{}'".format(self.compute_mean_uncertainty(), self.now))

            if new_test_point == None:
                print(self.node_id, "[[[[[[[[[[[ENDED]]]]]]]]]]]")
                # mqtt_client.close()
                self.lh.close()
                self.am_done = True
                # exit(0)

            changed = True
            print(self.node_id, "on to the next point", new_test_point)
            # self.hits = MyList()
            # self.snrs = list()
        else:
            new_test_point = [self.g_sf, self.g_cr, self.g_tx_power]
            print(self.node_id, "not enough certainty, continuing with the same point".format(abs(x_pos - x_neg)),
                  new_test_point)

        return new_test_point, changed

    def is_done(self):
        """
        Getter for am_done.
        Dictates if the PRR modeling phase is ended
        :return: self.am_done
        """
        return self.am_done

    def dr_to_sf(self, data_rate):
        """
        Converts from data_rate (0-5) to Spreading Factor (7-12)
        :param data_rate: The data rate at which the communication is taking place
        :return: The corresponding Spreading Factor as an integer from 7 to 12
        """
        return 7 + (5 - data_rate)

    def sf_to_dr(self, sf):
        """
        Converts from Spreading Factor (7-12) to data_rate (0-5)
        :param sf: The Spreading Factor at which the communication is taking place
        :return: The corresponding Data Rate
        """
        return 5 - (sf - 7)

    def fill_up_holes(self):
        """
        For those transmission configurations for which the PRR has not been computed, estimate it from the margins
        :return: None
        """
        for key, val in self.model.items():
            if np.isnan(val):
                self.model[key] = np.mean(self.margins[key][:2])

    def uplink_callback(self, msg, client):
        """
        Callback for each packet received
        :param msg: The message received
        :param client: The MQTT Client
        :return: None
        """
        assert self.node_id == msg.dev_id
        print(self.node_id, msg)
        self.lh.append_msg(str(msg))

        try:
            self.now = msg.metadata.time
        except:  # if, somehow, the MSG does not include such a field
            self.now = datetime.datetime.now().isoformat()

        if self.am_done:  # The PRR model is completed
            print(msg.dev_id, "Commanding node to stop as we are done")
            client.send(msg.dev_id, base64.b64encode(bytes([255, 0, 255])).decode('utf-8'), port=1, conf=False,
                        sched="replace")
            self.fill_up_holes()
            print(self.node_id, 'Sent')
            return

        paq_bytes = base64.b64decode(msg.payload_raw)
        if len(paq_bytes) == 3 and paq_bytes[0] == 0x00 and paq_bytes[1] == 0x01 and paq_bytes[2] == 0x02:
            # this kind of message is received when the node has not received any order yet
            # so let's comand it to start acquiring the PRR model
            print(msg.dev_id, "Switching mote to SWEEP mode with:")
            print(msg.dev_id,
                  "\n\tSF: {}\n\tCoding Rate: {}\n\tTX POWER: {}".format(self.g_sf, self.g_cr, self.g_tx_power))
            client.send(msg.dev_id, base64.b64encode(
                bytes([2, self.sf_to_dr(self.g_sf), self.g_cr, self.g_tx_power,  # now, fill it up to 9 bytes
                       0x00, 0x00, 0x00, 0x00, 0x00])).decode('utf-8'), port=1, conf=False,
                        sched="replace")
            print(self.node_id, 'Sent')

            return

        assert len(paq_bytes) == 5, "Incorrect number of bytes {}".format(len(paq_bytes))
        data_rate, coding_rate, tx_power, counter, req_ack = [int(v) for v in paq_bytes]
        spreading_factor = self.dr_to_sf(data_rate)

        packet_cr = int(msg.metadata.coding_rate[2]) - 4
        sf_str_pos = msg.metadata.data_rate.find('SF')
        bw_str_pos = msg.metadata.data_rate.find('BW')
        packet_sf = int(msg.metadata.data_rate[sf_str_pos + 2:bw_str_pos])
        if (spreading_factor != packet_sf or packet_cr != coding_rate) and packet_sf == 12 and (
                        packet_cr == 3 or packet_cr == 4):
            # this type of packet is received when the node has not received any message from me (GW) in 20 tranmissions
            # it is called a "bomb" because it is node's last resort
            self.lh.append_msg(
                "Node has sent a bomb for params = {}, {}, {}".format(spreading_factor, coding_rate, tx_power))
            print("Node has sent a bomb for params =", spreading_factor, coding_rate, tx_power)
            # self.bombed[(spreading_factor, coding_rate, tx_power)] = True

            # this is the first bomb for this specific parameters (last response may have been lost)
            if self.last_bomb_cr != coding_rate or \
                            self.last_bomb_sf != spreading_factor or \
                            self.last_bomb_tx_power != tx_power:
                self.last_bomb_cr = coding_rate
                self.last_bomb_sf = spreading_factor
                self.last_bomb_tx_power = tx_power

                if self.hits.get_params() != (spreading_factor, coding_rate, tx_power):
                    self.hits = MyList(spreading_factor, coding_rate, tx_power)
                    lost = max(0, counter - (-1))
                else:
                    lost = max(0, counter - self.last_counter)

                self.last_counter = counter
                m = self.hits.count(True)
                n = len(self.hits) + lost
                x_pos = self.get_pos_interval(m, n, self.confidence)
                x_neg = self.get_neg_interval(m, n, self.confidence)
                print(self.node_id, "m: {}, n: {}, certainty: {}".format(m, n, x_pos - x_neg))
                self.model[(spreading_factor, coding_rate, tx_power)] = m / n
                self.margins[(spreading_factor, coding_rate, tx_power)] = [x_neg, x_pos, x_pos - x_neg]

                self.bound_upper_rest(tx_power, spreading_factor, coding_rate, min(x_neg, m / n))
                self.bound_lower_rest(tx_power, spreading_factor, coding_rate, max(x_pos, m / n))

                pickle.dump(self.model, open('model_{}.p'.format(self.node_id), 'wb'))
                pickle.dump(self.margins, open('margins_{}.p'.format(self.node_id), 'wb'))
                self.lh.append_msg(
                    "Current uncertainty is: {}. Time = '{}'".format(self.compute_mean_uncertainty(), self.now))

                new_test_point = self.get_greater_margin()
                # partial saving :D
                pickle.dump(self.model, open('model_{}.p'.format(self.node_id), 'wb'))
                pickle.dump(self.margins, open('margins_{}.p'.format(self.node_id), 'wb'))
                self.lh.append_msg(
                    "Current uncertainty is: {}. Time = '{}'".format(self.compute_mean_uncertainty(), self.now))

                if new_test_point == None:
                    print(self.node_id, "[[[[[[[[[[[ENDED]]]]]]]]]]]")
                    self.lh.close()
                    self.am_done = True
                    return

                self.g_sf, self.g_cr, self.g_tx_power = new_test_point
                self.last_counter = -1
                self.hits = MyList(self.g_sf, self.g_cr, self.g_tx_power)  # spreading_factor, coding_rate, tx_power
                # self.snrs = list()
                print(
                    "Setting new params to SF: {}, CR: {}, TX_POWER: {}".format(self.g_sf, self.g_cr, self.g_tx_power))

            client.send(msg.dev_id,
                        base64.b64encode(bytes([self.sf_to_dr(self.g_sf), self.g_cr, self.g_tx_power])).decode('utf-8'),
                        conf=False,
                        port=1, sched="replace")
            return

        print(self.node_id, "Received uplink from", msg.dev_id, "with pkt counter", counter)

        if spreading_factor == 0 and data_rate == 0 and tx_power == 0:
            print(self.node_id, "Node did not received updating packet, sending it")
            client.send(msg.dev_id,
                        base64.b64encode(bytes([self.sf_to_dr(self.g_sf), self.g_cr, self.g_tx_power])).decode('utf-8'),
                        conf=False,
                        port=1, sched="replace")
            return

        if spreading_factor != self.g_sf or coding_rate != self.g_cr or tx_power != self.g_tx_power:
            print(self.node_id, "Last downlink got lost?")
            print(self.node_id,
                  "Globals: {}, {}, {} vs Received: {}, {}, {}".format(self.g_sf, self.g_cr, self.g_tx_power,
                                                                       spreading_factor, coding_rate, tx_power))

            client.send(msg.dev_id,
                        base64.b64encode(bytes([self.sf_to_dr(self.g_sf), self.g_cr, self.g_tx_power])).decode('utf-8'),
                        conf=False,
                        port=1, sched="replace")

            # _, _= update_params(hits, snrs) # gratuitous update
            # no se puede re-actualizar pq los g_sf, g_cr, y g_tx_power cambiaron con el anterior pkt, y los perdi =(
            return

        # first uplink for this transmission configuration
        if self.hits.get_params() == (None, None, None):
            self.hits.set_params(self.g_sf, self.g_cr, self.g_tx_power)

        if counter < self.last_counter:
            if (spreading_factor, coding_rate, tx_power) != self.hits.get_params():
                print("New config, reseting counters")
                self.hits = MyList(spreading_factor, coding_rate, tx_power)
                self.snrs = list()
                self.last_counter = -1
        else:
            assert (spreading_factor, coding_rate, tx_power) == self.hits.get_params()
            # if (spreading_factor, coding_rate, tx_power) != self.hits.get_params():
            #     print("WOT?")

        if counter == self.last_counter:
            print(self.node_id, "SAME COUNTER!!!")
            if hasattr(msg, 'is_retry') and msg.is_retry:
                print("MSG is retry")
            return

        # continua la recepcion normal
        missed_packets = counter - self.last_counter - 1
        print(self.node_id, missed_packets, 'lost packets')
        for _ in range(missed_packets):
            self.hits.append(False)
            self.snrs.append(np.nan)

        print(self.node_id, 'Normal hit :D')
        self.hits.append(True)

        if self.gw_id is not None:
            idx_my_gw = [g.gtw_id for g in msg.metadata.gateways].index(self.gw_id)
        else:
            idx_my_gw = 0

        self.snrs.append(msg.metadata.gateways[idx_my_gw].snr)
        self.last_counter = counter

        assert len(self.hits) < msg.counter + 2, 'This should not happen'

        # the idea is, after we have collected enough evidence, either test that the current certainty is enough
        # (in that case, proceed with the next config) or command the node to continue with the current config
        # (node should have news from the GW at least, every 20 packets or will send a bomb)
        if counter >= 10:
            new_test_point, changed = self.update_params()

            if self.am_done:
                print(msg.dev_id, "Commanding node to stop as we are done")
                client.send(msg.dev_id, base64.b64encode(bytes([255, 0, 255])).decode('utf-8'), port=1, conf=False,
                            sched="replace")
                print(self.node_id, 'Sent')
            else:
                if (changed is True) or (req_ack == 1):  # (len(self.hits) % 10 == 0 and counter > 10) or
                    self.g_sf = new_test_point[0]
                    self.g_cr = new_test_point[1]
                    self.g_tx_power = new_test_point[2]

                    print(self.node_id, "Sending updated config to node")
                    client.send(msg.dev_id,
                                base64.b64encode(bytes([self.sf_to_dr(self.g_sf), self.g_cr, self.g_tx_power])).decode(
                                    'utf-8'),
                                port=1,
                                conf=False,
                                sched="replace")
                    print(self.node_id, 'Sent')
                    # msg -> MSG(app_id, dev_id, hardware_serial, port, counter, payload_raw, payload_fields, metadata)

    def __init__(self, node_id, sfs, crs, tx_powers, certainty=0.2, gw_id='gateway_under_test', prev_model=None,
                 prev_margins=None, min_prr_testable=0, brute_force=False):
        """
        Worker sweep. His main duty is to construct the PRR model
        :param node_id: The string by which the node will be identified
        :param sfs: Set of Spreading Factors that will be swept
        :param crs: Set of Coding Rates that will be swept
        :param tx_powers: Set of Transmission Power values that will be swept
        :param certainty: The minimum level of certainty to be achieved (the difference between upper and lower bound)
        :param gw_id: The gateway id if only packets from it must be listened
        :param prev_model: Previous model or None if want to build it from scratch
        :param prev_margins: Previous margins or None if want to build it from scratch
        :param min_prr_testable: The minimum acceptable PRR. If we have some evidences that certain config will not achieve it, such a config will not be tested
        :param brute_force: Whether we are going with the brute force approach or the bounding technique
        """
        print("[Sweep] Initializing node handler for node:", node_id)

        self.node_id = node_id
        self.last_counter = -1
        self.hits = MyList(None, None, None)
        self.snrs = list()
        self.certainty = certainty
        self.confidence = 0.9  # confidence interval
        self.gw_id = gw_id
        self.min_prr_testable = min_prr_testable
        self.brute_force = brute_force
        self.g_tx_power = 14
        self.g_sf = 12  # SF= 10
        self.g_cr = 1  # CR = 4/7

        self.last_bomb_sf = None
        self.last_bomb_cr = None
        self.last_bomb_tx_power = None
        self.now = datetime.datetime.now().isoformat()

        assert self.g_tx_power in tx_powers
        assert self.g_sf in sfs
        assert self.g_cr in crs

        all_vars = list(product(sfs, crs, tx_powers))
        assert max(tx_powers) <= 14
        assert min(tx_powers) >= 2
        all_vars = sorted(all_vars, key=lambda x: (x[0] - 7) * 100 + abs(1.4 - int((x[2] - 2) / 4)) * 10 + x[1],
                          reverse=True)
        """
        Lo de arriba proyecta el espacio de variables a una dimension ordenable
        El primer dígito se lo lleva el SF -> lo mapeo de 7, 12 a (0, 5) * 100 (ocupa el digito más significativo)
        El segundo dígitoo se lo lleva el TXP -> lo mapeo de 2, 12 a (0, 1, 2, 4) * 10 y luego hago que el centro quede
        en 1.4. Asi, los primeros valores lo toman TXP14 y TXP2, luego TXP10 y TXP6
        El tercer digito se lo lleva el CR -> lo mapeo de 5, 8 a 5, 8 (no lo toco)
        """

        if prev_model is None:
            self.model = {v: np.nan for v in all_vars}
        else:
            assert isinstance(prev_model, OrderedDict)
            self.model = prev_model

        if prev_margins is None:
            self.margins = {v: [0, 1, 1] for v in all_vars}
        else:
            assert isinstance(prev_margins, OrderedDict)
            self.margins = prev_margins

        if prev_margins is not None or prev_model is not None:
            self.lh = LogHandler('log_{}.csv'.format(self.node_id), append=True)
        else:
            self.lh = LogHandler('log_{}.csv'.format(self.node_id))

        self.am_done = False

    def close(self):
        self.lh.close()


class MyList(list):
    """
    Extension of a list to also let us specify the config for which the PRR values will be stored
    """

    def __init__(self, sf, cr, txp):
        list.__init__(self)
        self.sf = sf
        self.cr = cr
        self.txp = txp

    def set_params(self, sf, cr, txp):
        self.sf = sf
        self.cr = cr
        self.txp = txp

    def get_params(self):
        return (self.sf, self.cr, self.txp)

    def __repr__(self):
        return "SF: {}, CR: {}, TXP: {}\n".format(self.sf, self.cr, self.txp) + list.__repr__(self)
