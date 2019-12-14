from network import LoRa
import socket
import time
import binascii
import machine
import pycom
import struct
import math

hash_table = [247, 146, 42, 23, 21, 143, 201, 47, 100, 80, 12, 153, 58, 34, 238, 123, 140, 61, 213, 43, 190, 110, 90, 35, 105, 250, 185, 73, 148, 230, 65, 186, 28, 138, 149, 31, 166, 189, 164, 122, 227, 204, 6, 91, 96, 69, 60, 3, 217, 32, 158, 40, 178, 89, 173, 53, 253, 55, 126, 248, 10, 205, 11, 79, 160, 52, 85, 133, 182, 54, 27, 214, 107, 243, 22, 120, 194, 193, 63, 95, 246, 226, 224, 239, 169, 241, 74, 180, 118, 234, 4, 30, 168, 221, 183, 231, 177, 41, 25, 176, 191, 171, 119, 56, 59, 152, 78, 19, 46, 172, 156, 18, 159, 103, 141, 161, 136, 170, 157, 9, 8, 97, 86, 255, 134, 39, 174, 16, 151, 49, 220, 66, 117, 233, 132, 162, 14, 196, 135, 36, 81, 45, 165, 38, 20, 116, 232, 223, 29, 76, 219, 137, 121, 203, 145, 115, 129, 245, 202, 142, 57, 198, 62, 84, 50, 75, 167, 98, 197, 154, 51, 225, 242, 207, 184, 2, 244, 155, 228, 150, 163, 210, 94, 83, 249, 195, 240, 104, 187, 237, 109, 5, 68, 15, 229, 209, 181, 236, 215, 211, 33, 92, 93, 127, 139, 208, 130, 252, 17, 188, 216, 131, 101, 67, 144, 71, 77, 112, 87, 179, 7, 114, 99, 235, 212, 44, 26, 175, 200, 48, 251, 113, 1, 102, 82, 192, 147, 111, 199, 124, 70, 218, 128, 64, 125, 24, 108, 88, 222, 37, 206, 0, 72, 13, 254, 106]


def factor():
    pycom.nvs_set('order', 0x00)
    pycom.nvs_set('hash', 0x00)
    return

def two_bytes_to_float(two_bytes):
    assert len(two_bytes) == 2
    raw = struct.unpack('H', two_bytes)
    raw = HalfToFloat(raw[0])
    return struct.unpack('f', struct.pack('I', raw))[0]

# http://forums.devshed.com/python-programming-11/converting-half-precision-floating-hexidecimal-decimal-576842.html
def HalfToFloat(h):
    s = int((h >> 15) & 0x00000001)    # sign
    e = int((h >> 10) & 0x0000001f)    # exponent
    f = int(h & 0x000003ff)            # fraction

    if e == 0:
       if f == 0:
          return int(s << 31)
       else:
          while not (f & 0x00000400):
             f <<= 1
             e -= 1
          e += 1
          f &= ~0x00000400
          print(s,e,f)
    elif e == 31:
       if f == 0:
          return int((s << 31) | 0x7f800000)
       else:
          return int((s << 31) | 0x7f800000 | (f << 13))

    e = e + (127 -15)
    f = f << 13

    return int((s << 31) | (e << 23) | f)


def compute_over_the_air_time(payload_length, sf, cr):
    # lora.stats()[7] / 1000 # <- alternativamente
    payload_length += 13
    BW = 125e3
    preamble_symbols = 8
    header_length = 0
    explicit_header = 0

    if sf == 0 and cr == 0:
        return 0

    assert 7 <= sf <= 12
    assert 5 <= cr <= 8
    de = 1 if sf >= 11 else 0
    # http://forum.thethingsnetwork.org/t/spreadsheet-for-lora-airtime-calculation/1190/15
    t_sym = pow(2, sf) / BW * 1000  # symbol time in ms
    t_preamble = (preamble_symbols + 4.25) * t_sym  # over the air time of the preamble
    payload_symbol_number = 8 + max([(math.ceil(
            (8 * (payload_length + header_length) - 4 * sf + 28 + 16 - 20 * (1 - explicit_header)) / (
                4 * (sf - 2 * de))) * cr), 0])  # number of symbols of the payload
    t_payload = payload_symbol_number * t_sym  # payload time in ms
    t_packet = t_preamble + t_payload
    return t_packet / 1000  # expressed in seconds

def craft_packet(tx_power, data_rate, coding_rate, counter, req_ack):
    assert 0 <= counter <= 255
    paq_bytes = bytes([data_rate, coding_rate, tx_power, counter & 0xff, req_ack])
    return paq_bytes


# Pearson hashing
def compute_hash(message):
    hash = len(message) % 256
    for i in message:
        hash = hash_table[(hash+i) % 256]

    return hash

def fill_bytes(counter, hash, num_bytes):
    assert isinstance(num_bytes, int)
    assert 2 < num_bytes <= 230
    counter = counter % 256
    return bytes([counter]) + bytes([hash]) + struct.pack('B'* (num_bytes - 2), *range(num_bytes - 2))

def accum_vector(vector):
    v = 0
    ret = []
    for _ in range(len(vector)):
        v += vector[_]
        ret.append(v)

    assert abs(ret[-1] - 1) < 1e-3
    ret[-1] = 1 # force 1
    return ret

def get_dr_cr_from_index(index):
    tx_power = [2, 6, 10, 14][index // 12]
    cr = 3 if ((index % 12) // 6) else 1
    dr = 12 - ((index % 6) + 7)
    return dr, cr, tx_power

def get_config_from_vector(config):
    r = get_random_0_1()
    acc_config = accum_vector(config)

    cfg = 0
    while r > acc_config[cfg]:
        cfg += 1

    return cfg

def get_config_from_accum_vector(acc_config):
    r = get_random_0_1()
    print("Random number", r)

    cfg = 0
    while r > acc_config[cfg]:
        cfg += 1

    return cfg

def get_random_0_1():
    return (machine.rng() % 16777215) / 16777215

def get_inter_arrival(lambda_):
    return - math.log(1.0 - get_random_0_1()) / lambda_

def create_lora_adr(handler, app_eui, app_key):
    lora = LoRa(mode=LoRa.LORAWAN, region=LoRa.EU868, adr = False)
    lora.callback(trigger=LoRa.RX_PACKET_EVENT, handler=handler)
    lora.join(activation=LoRa.OTAA, auth=(app_eui, app_key), timeout=0, dr=0)

    while not lora.has_joined():
        time.sleep(2.5)
        print("Waiting for lora to join")

    return lora

def create_lora(handler, app_eui, app_key, tx_power = 14, coding_rate = 1):
    assert 2 <= tx_power <= 14
    assert 1 <= coding_rate <= 4

    lora = LoRa(mode=LoRa.LORAWAN, region=LoRa.EU868, adr = False)
    lora.tx_power(tx_power)    # from 2 to 14
    lora.coding_rate(coding_rate) # 1 = 4/5, 2 = 4/6, 3 = 4/7, 4 = 4/8
    lora.callback(trigger=LoRa.RX_PACKET_EVENT, handler=handler)
    lora.join(activation=LoRa.OTAA, auth=(app_eui, app_key), timeout=0, dr=0)

    while not lora.has_joined():
        time.sleep(2.5)
        print("Waiting for lora to join")

    return lora


def create_lora_abp(tx_power = 14, coding_rate = 1):
    assert 2 <= tx_power <= 14
    assert 1 <= coding_rate <= 4

    lora = LoRa(mode=LoRa.LORAWAN, region=LoRa.EU868, adr = False)
    lora.tx_power(tx_power)    # from 2 to 14
    lora.coding_rate(coding_rate) # 1 = 4/5, 2 = 4/6, 3 = 4/7, 4 = 4/8
    lora.callback(trigger=LoRa.RX_PACKET_EVENT, handler=update_tx_params_cb)
    lora.join(activation=LoRa.ABP, auth=(dev_addr, nwk_swkey, app_swkey))
    return lora


def create_socket_adr():
    s = socket.socket(socket.AF_LORA, socket.SOCK_RAW)
    s.setsockopt(socket.SOL_LORA, socket.SO_CONFIRMED, False)
    s.setblocking(True)
    return s

def create_socket(data_rate = 5):
    assert 0 <= data_rate <= 5
    s = socket.socket(socket.AF_LORA, socket.SOCK_RAW)
    # SF7   (data rate = 5)
    # SF12  (data rate = 0)
    s.setsockopt(socket.SOL_LORA, socket.SO_DR, data_rate)
    s.setsockopt(socket.SOL_LORA, socket.SO_CONFIRMED, False)
    s.setblocking(True)
    return s

def set_tx_power(lora, tx_power = 14):
    global g_tx_power
    assert 2 <= tx_power <= 14
    lora.tx_power(tx_power)
    # lora.join(activation=LoRa.ABP, auth=(dev_addr, nwk_swkey, app_swkey))
    g_tx_power = tx_power

def set_coding_rate(lora, coding_rate = 1):
    global g_coding_rate
    assert 1 <= coding_rate <= 4
    lora.coding_rate(coding_rate)
    # lora.join(activation=LoRa.ABP, auth=(dev_addr, nwk_swkey, app_swkey))
    g_coding_rate = coding_rate

def set_data_rate(s, lora, data_rate = 5):
    global g_data_rate
    assert 0 <= data_rate <= 5
    s.setsockopt(socket.SOL_LORA, socket.SO_DR, data_rate)
    # lora.tx_power(g_tx_power)   # dirty trick
    # lora.join(activation=LoRa.ABP, auth=(dev_addr, nwk_swkey, app_swkey))   # no necesario pero para reiniciar el counter
    g_data_rate = data_rate

def data_rate_to_sf(data_rate):
    return (5 - data_rate) + 7

def coding_rate_to_cr(coding_rate):
    return coding_rate + 4

def get_off_period(payload_length = 4):
    sending_time = 3
    return max((
        compute_over_the_air_time(payload_length,
            data_rate_to_sf(g_data_rate),
            coding_rate_to_cr(g_coding_rate)) / 0.01) - sending_time,
        0)
