import time
from multiprocessing import Process
from threading import Timer, Thread
import pickle
import numpy as np
import ttn
import base64
import warnings

from gym_lora import LoRaWorld, reduce_to
from worker_normal import Worker as Worker_n
from worker_sweep import Worker as Worker_s



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Bootstrapper():
    def __init__(self):
        self.p = None

        self.app_id = "app-prueba-2"
        self.access_key = "" # removed before uploading it to github

        self.app = ttn.ApplicationClient(self.app_id, self.access_key, handler_address="", cert_content="",
                                         discovery_address="discovery.thethings.network:1900")

        ttn_handler = ttn.HandlerClient(self.app_id, self.access_key)
        self.mqtt_client = ttn_handler.data()
        self.mqtt_client.set_uplink_callback(self.meta_handler)
        print("All set to factor devices")

    def meta_handler(self, msg, client):
        paq_bytes = base64.b64decode(msg.payload_raw)
        if len(paq_bytes) == 3 and paq_bytes[0] == 0x00 and paq_bytes[1] == 0x01 and paq_bytes[2] == 0x02:
            print("Node {} already factored".format(msg.dev_id))
        else:
            client.send(msg.dev_id, base64.b64encode(bytes([1, 0, 0, 0, 0, 0])).decode('utf-8'), port=1, conf=False,
                        sched="replace")
            print("Factoring device", msg.dev_id)

    def run(self):
        self.mqtt_client.connect()
        time.sleep(60 * 60 * 24 * 4)  # 4 days


if __name__ == '__main__':
    bs = Bootstrapper()
    bs.run()
