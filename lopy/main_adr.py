from network import LoRa
import socket
import time
import binascii
import machine
import pycom
import struct
import math
from helper import *

def packet_received_hd(lora):
    global g_tx_power, g_data_rate, g_coding_rate
    global paq_bytes, order
    global g_off_period, g_counter, g_keep_alive_counter, g_done

    events = lora.events()
    print("Packet received from gateway")
    if events & LoRa.RX_PACKET_EVENT:
        recv_bytes = s.recv(64)
        if len(recv_bytes) == 3:
            g_keep_alive_counter = 0
            data_rate, coding_rate, tx_power = recv_bytes
            print("Received response from beacon")
            print(data_rate, coding_rate, tx_power)

            if data_rate == 255 and coding_rate == 0 and tx_power == 255:
                print("received signal to stop! I am done!")
                g_done = True
                return

            if 2 <= tx_power <= 14:
                if tx_power != g_tx_power:
                    set_tx_power(lora, tx_power)
                    g_tx_power = tx_power
                    print("new tx_power", tx_power)
                    g_counter = -1
                else:
                    print("Same tx power")
            else:
                print("tx_power exceeded limtis", tx_power)

            if 0 <= data_rate <= 5:
                if data_rate != g_data_rate:
                    set_data_rate(s, lora, data_rate)
                    g_data_rate = data_rate
                    print("new data_rate", data_rate)
                    g_counter = -1
                else:
                    print("Same data rate")
            else:
                print("data_rate exceeded limits", data_rate)

            if 1 <= coding_rate <= 4:
                if coding_rate != g_coding_rate:
                    set_coding_rate(lora, coding_rate)
                    g_coding_rate = coding_rate
                    print("new coding_rate", coding_rate)
                    g_counter = -1
                else:
                    print("Same coding rate")
            else:
                print("coding_rate exceeded limits", coding_rate)

            g_off_period = get_off_period()
        elif len(recv_bytes) >= 4:
            # new order received
            if recv_bytes[0] == 0x01:   # clear order
                 pycom.nvs_set('order', 0x00)
                 pycom.nvs_set('hash', 0x00)
                 machine.deepsleep(120000)   # deep sleep for two minutes
            elif recv_bytes[0] == 0x02: # param sweep
                # #1 byte starting_data_rate
                # #2 byte starting_cr
                # #3 byte starting tx_power

                g_data_rate_t = recv_bytes[1]
                g_coding_rate_t = recv_bytes[2]
                g_tx_power_t = recv_bytes[3]

                if g_data_rate_t == g_data_rate and g_coding_rate_t == g_coding_rate and g_tx_power_t == g_tx_power and order == 0x02:
                    print("Duplicate sweep order")
                    return

                g_data_rate = g_data_rate_t
                g_coding_rate = g_coding_rate_t
                g_tx_power = g_tx_power_t

                assert 0 <= g_data_rate <= 5
                assert 1 <= g_coding_rate <= 4
                assert 2 <= g_tx_power <= 14

                print("Received order SWEEP from GW.")
                print("\tData rate: {}.\n\tCoding rate: {}.\n\tTX power: {}.".format(g_data_rate, g_coding_rate, g_tx_power))

                pycom.nvs_set('data_rate', g_data_rate)
                pycom.nvs_set('coding_rate', g_coding_rate)
                pycom.nvs_set('tx_power', g_tx_power)
                pycom.nvs_set('order', 0x02)    # after the asserts and sets

                time.sleep(0.2)
                machine.deepsleep(5000)
            elif recv_bytes[0] == 0x03: # normal transmission mode
                g_tx_power = recv_bytes[1]
                g_pkt_length = recv_bytes[2]
                g_lambda_ = struct.unpack('!f', recv_bytes[3:7])[0]  # in events per second
                g_lambda_int = struct.unpack('i', recv_bytes[3:7])[0]

                leftover = len(recv_bytes) - 7
                assert leftover % 3 == 0
                num_confs = int(leftover / 3)

                computed_hash = compute_hash(recv_bytes)
                hash_rcv = pycom.nvs_get('hash')

                if computed_hash == hash_rcv:
                    print("Skipping duplicate with:", computed_hash)
                    return
                else:
                    print("Non-duplicated packet with hash:", computed_hash)
                    pycom.nvs_set('hash', computed_hash)

                print("Received order NORMAL from GW.")
                print("\tTX power: {}\n\tPKT length: {}\n\tLambda: {}".format(g_tx_power, g_pkt_length, g_lambda_))

                pycom.nvs_set('cfgs', num_confs)
                for n_conf in range(num_confs):
                    print("Parsing config #{}".format(n_conf + 1))

                    config_id = int(recv_bytes[7 + n_conf * 3 + 0])
                    config_freq = two_bytes_to_float(recv_bytes[7 + n_conf * 3 + 1 : 7 + n_conf * 3 + 3])
                    assert 0 <= config_id < 48
                    assert 0 < config_freq <= 1

                    pycom.nvs_set('cfg_' + str(n_conf), config_id)
                    pycom.nvs_set('cfg_f_' + str(n_conf), struct.unpack('i', struct.pack('!f', config_freq))[0])
                    print("Result of parsing:\t\t config[{}] = {}".format(config_id, config_freq))

                assert 2 <= g_tx_power <= 14
                assert 2 <= g_pkt_length <= 250
                assert 0 <= g_lambda_ <= 1  # events per

                pycom.nvs_set('tx_power', g_tx_power)
                pycom.nvs_set('pkt_length', g_pkt_length)
                pycom.nvs_set('lambda', g_lambda_int)
                pycom.nvs_set('order', 0x03)

                time.sleep(0.2)
                machine.deepsleep(5000)
            else:
                print("Unrecognised order received:", int(recv_bytes[0]))

            return
        else:
            print("received packet format unrecognised")


pycom.heartbeat(True)
# dev_addr = struct.unpack(">l", binascii.unhexlify(''))[0]
# nwk_swkey = binascii.unhexlify("")	# removed before uploading it to github
# app_swkey = binascii.unhexlify("")	# removed before uploading it to github
app_eui = binascii.unhexlify('')	# removed before uploading it to github
app_key = binascii.unhexlify('')	# removed before uploading it to github

g_tx_power = 14
g_data_rate = 0 # sf = 12
g_coding_rate = 1
g_lambda_ = 1
g_pkt_length = 10
paq_bytes = None
g_counter = -1
g_keep_alive_counter = 0

lora = LoRa(mode=LoRa.LORAWAN, region=LoRa.EU868, adr = False)
lora.nvram_restore()
if lora.has_joined():
    print('Joining not required')
    lora.callback(trigger=LoRa.RX_PACKET_EVENT, handler=packet_received_hd)
    lora.nvram_save()
else:
    print('Joining for the first time')
    lora = create_lora_adr(handler = packet_received_hd,
                        app_eui = app_eui,
                        app_key = app_key)
    lora.nvram_save()


for i in range(16):
    lora.remove_channel(i)
    print('Removed channel {}'.format(i))

s = create_socket_adr()

order = pycom.nvs_get('order')
if order is None or order == 0x00 or order == 0x02:
    print("Stored order is none")
    set_tx_power(lora, tx_power = 14)
    set_coding_rate(lora, coding_rate = 1)
    set_data_rate(s, lora, data_rate = 0)

    while (order is None or order == 0x00):
        print("Beeping to get a new order")
        s.send(bytes([0x00, 0x01, 0x02]))   # means, "gimme an order"
        time.sleep(30)
        order = pycom.nvs_get('order')

elif order == 0x03:
    print("Stored order is normal transmission")

    g_tx_power = pycom.nvs_get('tx_power')
    g_pkt_length = pycom.nvs_get('pkt_length')
    lambda_int = pycom.nvs_get('lambda')
    num_confs = pycom.nvs_get('cfgs')
    hash = pycom.nvs_get('hash')

    assert hash is not None
    assert g_tx_power is not None
    assert g_pkt_length is not None
    assert lambda_int is not None
    assert 0 < num_confs < 8

    tx_confs = [0] * 48
    for conf in range(num_confs):
        conf_idx = pycom.nvs_get('cfg_' + str(conf))
        conf_freq_int = pycom.nvs_get('cfg_f_' + str(conf))
        conf_freq = struct.unpack('!f', struct.pack('i', conf_freq_int))[0]

        assert 0 <= conf_idx < 48
        assert 0.05 <= conf_freq <= 1
        tx_confs[conf_idx] = conf_freq
        print("Recovered Conf: {}, Freq: {}".format(conf_idx, conf_freq))

    g_lambda_ = struct.unpack('!f', struct.pack('i', lambda_int))[0]
    assert 0 <= g_lambda_ <= 1
    assert 2 <= g_pkt_length <= 250

    print("TX power: {}\nPKT length: {}\nRate: {}".format(
        g_tx_power, g_pkt_length, g_lambda_
    ))

    acc_config = accum_vector(tx_confs)

    while True:
        g_counter += 1
        paq_bytes = fill_bytes(g_counter, hash, g_pkt_length)

        print("sending packet")
        pycom.heartbeat(False)
        pycom.rgbled(0x7f0000)
        s.send(paq_bytes)
        pycom.heartbeat(True)

        next_ = max(0, get_inter_arrival(g_lambda_) - 3)
        print("Sleeping for", next_, "seconds")
        if next_ > 0:
            time.sleep(next_)

else:
    print("Unrecognised order in NVS memory:", int(order))

print("Ended, reseting")
machine.reset()
