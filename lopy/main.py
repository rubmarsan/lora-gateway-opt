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
    events = lora.events()
    if events & LoRa.RX_PACKET_EVENT:
        print("Packet received from gateway")

app_eui = binascii.unhexlify('')	# removed before uploading it to github
app_key = binascii.unhexlify('')	# removed before uploading it to github
g_tx_power = 14
g_coding_rate = 1
g_data_rate = 0

lora = LoRa(mode=LoRa.LORAWAN, region=LoRa.EU868, adr = False)
lora.nvram_restore()
if lora.has_joined():
    print('Joining not required')
    lora.callback(trigger=LoRa.RX_PACKET_EVENT, handler=packet_received_hd)
    lora.nvram_save()
else:
    print('Joining for the first time')
    lora = create_lora(handler = packet_received_hd,
                        app_eui = app_eui,
                        app_key = app_key,
                        tx_power = g_tx_power,
                        coding_rate = g_coding_rate)
    lora.nvram_save()

s = create_socket(data_rate = g_data_rate)

while True:
    print("Beeping")
    s.send(bytes([0x00, 0x01, 0x02]))   # means, "gimme an order"
    time.sleep(5)
