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
    return

pycom.heartbeat(False)
pycom.wifi_on_boot(False)


# dev_addr = struct.unpack(">l", binascii.unhexlify(''))[0] 	# removed before uploading it to github
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

s = create_socket_adr()

while True:
    set_tx_power(lora, tx_power = 14)
    s.send(bytes([0x01, 0x02, 0x03, 0x04, 0x05]))
    time.sleep(3)
