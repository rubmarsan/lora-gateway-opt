import base64
import struct

import numpy as np

from loghandler import LogHandler


class Worker():
    """
    Worker normal, its main duty is to command nodes to send packets under a specific configuration
    """

    # This table is used by the Pearson hashing algorithm
    hash_table = [247, 146, 42, 23, 21, 143, 201, 47, 100, 80, 12, 153, 58, 34, 238, 123, 140, 61, 213, 43, 190, 110,
                  90, 35, 105, 250, 185, 73, 148, 230, 65, 186, 28, 138, 149, 31, 166, 189, 164, 122, 227, 204, 6, 91,
                  96, 69, 60, 3, 217, 32, 158, 40, 178, 89, 173, 53, 253, 55, 126, 248, 10, 205, 11, 79, 160, 52, 85,
                  133, 182, 54, 27, 214, 107, 243, 22, 120, 194, 193, 63, 95, 246, 226, 224, 239, 169, 241, 74, 180,
                  118, 234, 4, 30, 168, 221, 183, 231, 177, 41, 25, 176, 191, 171, 119, 56, 59, 152, 78, 19, 46, 172,
                  156, 18, 159, 103, 141, 161, 136, 170, 157, 9, 8, 97, 86, 255, 134, 39, 174, 16, 151, 49, 220, 66,
                  117, 233, 132, 162, 14, 196, 135, 36, 81, 45, 165, 38, 20, 116, 232, 223, 29, 76, 219, 137, 121, 203,
                  145, 115, 129, 245, 202, 142, 57, 198, 62, 84, 50, 75, 167, 98, 197, 154, 51, 225, 242, 207, 184, 2,
                  244, 155, 228, 150, 163, 210, 94, 83, 249, 195, 240, 104, 187, 237, 109, 5, 68, 15, 229, 209, 181,
                  236, 215, 211, 33, 92, 93, 127, 139, 208, 130, 252, 17, 188, 216, 131, 101, 67, 144, 71, 77, 112, 87,
                  179, 7, 114, 99, 235, 212, 44, 26, 175, 200, 48, 251, 113, 1, 102, 82, 192, 147, 111, 199, 124, 70,
                  218, 128, 64, 125, 24, 108, 88, 222, 37, 206, 0, 72, 13, 254, 106]

    def compute_hash(self, message):
        """
        Implementation of the Pearson Hashing. See https://en.wikipedia.org/wiki/Pearson_hashing
        :param message: The byte array encoding the payload of the message
        :return: The 8-bit hash
        """
        hash = len(message) % 256
        for i in message:
            hash = self.hash_table[(hash + i) % 256]

        return hash

    def is_done(self):
        """
        Returns whether we have ended
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

    @staticmethod
    def compress_config_vector(vector):
        """
        Compress a dense vector in a sparse vector by selecting the configs larger than 5% and zero-ing the rest.
        After it the vector is normalized so as to represent a discrete PDF
        :param vector: A dense vector with the probability for each configuration
        :return: A sparse vector with configs greater than 0.05%
        """
        vector_16 = vector.astype(np.float16)

        vector_16[vector_16 < 0.05] = 0
        vector_16 /= vector_16.sum()
        idxs = np.where(vector_16 > 0)[0]

        return idxs, vector_16[idxs]

    def get_updated_config(self):
        """
        Construct the byte array to be sent to the node. It contains the configuration that such a node must have.
        The configuration includes the new pkt_length, the rate at which packets should be sent and the probability
        of using each configuration.
        :return: The byte-array with the new config
        """
        rate_bytes = struct.pack("!f", self.g_rate)
        assert (len(rate_bytes)) == 4

        # the 3 is the indicator of a "normal operation" (as opossed to the 2 of a "sweep operation")
        # technically, g_tx_power should not be sent, but it is here as a part of legacy code :[
        ret = bytes([3, self.g_tx_power, self.g_pkt_length]) + rate_bytes

        config_ids, config_vals = self.compress_config_vector(self.config_vector)
        for i in range(len(config_ids)):
            ret += bytes([config_ids[i]]) + struct.pack('<e', config_vals[i])  # np.float16(config_vals[i]).tostring()

        assert (len(ret) - 7) % 3 == 0
        return ret

    def update_config(self, new_rate=None, new_config_vector=None):
        """
        A method to receive the new config computed by the main thread. When possible, this new config will be sent
        to the corresponding node (note that each worker_normal thread is assigned to each IoT node).
        :param new_rate: the new \lambda at which packets should be sent
        :param new_config_vector: the new vector indicating the probability of choosing each config
        :return: None
        """
        self.lh.append_msg(self.node_id + " Updating config: " + str(new_rate) + " " + str(new_config_vector))
        if new_rate is not None:
            self.g_rate = new_rate
            self.force_update = True
            print(self.node_id, "Forcing next update", self.force_update, str(self))

        if new_config_vector is not None:
            self.config_vector = new_config_vector
            self.force_update = True
            print(self.node_id, "Forcing next update", self.force_update, str(self))
        # setattr(self, 'force_update', False) is not None
        return

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

        # print("-------->", self.node_id, "next update?", self.force_update, str(self))

        if self.am_done:
            print(msg.dev_id, "Commanding node to stop as we are done")
            client.send(msg.dev_id, base64.b64encode(bytes([255, 0, 255])).decode('utf-8'), port=1, conf=False,
                        sched="replace")
            print(self.node_id, 'Sent')
            return

        paq_bytes = base64.b64decode(msg.payload_raw)
        if (len(paq_bytes) == 3 and paq_bytes[0] == 0x00 and paq_bytes[1] == 0x01 and paq_bytes[2] == 0x02) or \
                self.force_update:
            # this kind of message is received when the node has not received any order yet or its config must be updated
            # so let's comand it to start sending packets

            if not self.force_update:
                print(msg.dev_id, "Switching mote to NORMAL mode with:")
            else:
                print("Updating was forced")

            print(msg.dev_id, "\n\tTX POWER: {}\n\tRate: {}".format(self.g_tx_power, self.g_rate))
            print("and config vector: ")
            config_ids, config_vals = self.compress_config_vector(self.config_vector)
            for i in range(len(config_ids)):
                print("\t\tConfig {} percentage {}".format(config_ids[i], config_vals[i]))

            new_config = self.get_updated_config()
            client.send(msg.dev_id, base64.b64encode(new_config).decode('utf-8'), port=1, conf=False,
                        sched="replace")
            print(self.node_id, 'Sent')
            self.hash = self.compute_hash(new_config)

            if self.force_update:
                self.force_update = False

            return

        counter = paq_bytes[0]
        rcv_hash = paq_bytes[1]

        if counter < self.last_counter:
            if abs(counter - self.last_counter) > 5 and counter < 5:
                print(self.node_id, "New config, reseting counters")
                self.last_counter = -1
            else:
                print(self.node_id, "-----MASIVE DISORDER!-------")
                if len(self.hits) > counter and self.hits[counter] is False:
                    print(self.node_id, "Intentando arreglar, he fijado {} a true... cuando era... {}".format(counter,
                                                                                                              self.hits[
                                                                                                                  counter]))
                    self.hits[counter] = True

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

        # print(self.node_id, 'Normal hit :D')
        self.hits.append(True)

        if self.gw_id is not None:
            idx_my_gw = [g.gtw_id for g in msg.metadata.gateways].index(self.gw_id)
        else:
            idx_my_gw = 0


        if self.hash is None:
            print("-----[I do not have any stored hash. Accepting received hash]----")
            self.hash = rcv_hash

        if rcv_hash != self.hash or len(paq_bytes) != self.g_pkt_length:
            print("Hash/Length mismatch, updating node config")
            print("and config vector: ")
            config_ids, config_vals = self.compress_config_vector(self.config_vector)
            for i in range(len(config_ids)):
                print("\t\tConfig {} percentage {}".format(config_ids[i], config_vals[i]))

            new_config = self.get_updated_config()
            client.send(msg.dev_id, base64.b64encode(new_config).decode('utf-8'), port=1, conf=False,
                        sched="replace")
            print(self.node_id, 'Sent')
            self.hash = self.compute_hash(new_config)
            # self.last_counter = -1

        self.snrs.append(msg.metadata.gateways[idx_my_gw].snr)
        self.last_counter = counter

        print(self.node_id,
              "PRR SO FAR: {} ({}/{})".format(self.hits.count(True) / len(self.hits), self.hits.count(True),
                                              len(self.hits)))
        self.lh.append_msg(
            "PRR SO FAR: {} ({}/{})".format(self.hits.count(True) / len(self.hits), self.hits.count(True),
                                            len(self.hits)))
        self.lh.file.flush()

    def __init__(self, node_id, config_vector, tx_power, pkt_length, rate, gw_id='gateway_under_test'):
        """
        Worker normal, its main duty is to command nodes to send packets under a specific configuration
        :param node_id: The string by which the node will be identified
        :param config_vector: The vector indicating the probability of choosing one specific config
        :param tx_power: The transmission power to use.
        :param pkt_length: The length of the packets to be sent.
        :param rate: The rate at which packets should be sent.
        :param gw_id: gw_id: The gateway id if only packets from it must be listened
        """

        print("[Normal] Initializing node handler for node:", node_id)
        self.node_id = node_id
        self.last_counter = -1
        self.hits = list()
        self.snrs = list()
        self.gw_id = gw_id
        self.force_update = False
        self.hash = None

        self.g_tx_power = tx_power
        self.config_vector = config_vector
        self.g_pkt_length = pkt_length
        self.g_rate = float(rate)

        assert abs(self.config_vector.sum() - 1) < 1e-3
        assert np.all(self.config_vector <= 1)
        assert np.all(self.config_vector >= 0)
        assert self.g_tx_power in range(2, 15)
        assert 1 <= self.g_pkt_length <= 220
        assert 0 <= self.g_rate <= 1

        self.lh = LogHandler('log_normal_{}.csv'.format(self.node_id))
        self.lh.append_msg(self.node_id + " Base config: " + str(rate) + " " + str(config_vector))
        self.am_done = False

        # new_config = self.get_updated_config()
        # self.hash = self.compute_hash(new_config)

    def close(self):
        self.lh.close()
