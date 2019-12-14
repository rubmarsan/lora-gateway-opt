import time
from threading import Timer

import numpy as np
import ttn

from gym_lora import LoRaWorld, reduce_to
from worker_normal import Worker as Worker_n
from worker_sweep import Worker as Worker_s


class bcolors:
    """
    Used to print in the console (linux) with color
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Bootstrapper():
    """
    This is the class in charge of bootstrapping the system. Either to construct the PRR model or to work in normal mode
    """

    def __init__(self):
        # for constructing the PRR model we need to specify the desired params to sweep
        sfs = [7, 8, 9, 10, 11, 12]
        crs = [1]
        tx_powers = [2, 6, 10, 14]

        certainty = 0.15  # the desired difference between the upper and lower bound of the confidence intervals
        self.p = None
        self.app_id = "app-prueba-2"
        self.access_key = ""	# removed before uploading it to github
        self.app = ttn.ApplicationClient(self.app_id, self.access_key, handler_address="", cert_content="",
                                         discovery_address="discovery.thethings.network:1900")
        mode = "normal"  # modes:
        # "sweep" to construct the PRR model
        # "normal" to operate in normal mode

        if mode == "normal":

            np.random.seed(7)
            self.num_nodos = 5
            self.lambdas_ = 1 / np.array([60, 130, 100, 150, 120])  # modeled as the inverse of inter-packet time
            self.lengths = np.array([15 * 2, 13 * 2, 18 * 2, 12 * 2, 220])  # in bytes
            self.priorities = np.array([1, 1, 1, 1, 0])  # all nodes have the same priority
            self.snrs = np.array([6, 6, 6, 6, 6])  # the SNR of the nodes
            self.forced_nodes = [4]  # this indicates that the 5th node (4th in zero-index system) is
            # not a part of our network and thus, its configuration is forced
            self.forced_configs = np.zeros((49, self.lambdas_.shape[0]))
            self.forced_configs[43, 4] = 1  # the interfering is jamming the SF7 CR4/5 TXP14 config

            self.model_paths = ['raspi2/model_cambridge_1.p', 'raspi2/model_cambridge_1.p',
                                'raspi2/model_cambridge_1.p',
                                'raspi2/model_cambridge_1.p', 'raspi2/model_cambridge_1.p']
            # this is where we specify the PRR models obtained in the "sweep" mode

            self.acceptable_tx_powers = np.ones((4, self.num_nodos))  # all configurations are permitted
            self.C_opt = np.zeros((49, self.lambdas_.shape[0]))
            self.re_compute()

            current_performance = self.lora.compute_network_performance()
            self.last_update = time.time()
            print("Network performance: ", current_performance)

            self.handlers = dict()
            # register the handlers so the packets are enrouted to the appropriate representation of nodes
            for i in range(5):
                self.handlers['cambridge_{}'.format(i + 1)] = Worker_n('cambridge_{}'.format(i + 1),
                                                                       self.lora.C_opt[1:, i],
                                                                       tx_power=14, pkt_length=self.lengths[i],
                                                                       rate=self.lambdas_[i],
                                                                       gw_id='gateway_under_test')

            # if the fifth node is present, let's consider it is going to start jamming after an hour
            if 'cambridge_5' in self.handlers and isinstance(self.handlers['cambridge_5'], Worker_n):
                t = Timer(60 * 60, self.change_pattern_interferer)
                t.start()
                print("Timer set to change cambridge_5 config after an hour")
            else:
                print("Did not set the timer as cambridge_5 is not present")


        elif mode == "sweep":
            # sweep mode, let's create the handler for each device
            devices = list(self.app.devices())
            self.handlers = {
            dev.dev_id: Worker_s(dev.dev_id, sfs, crs, tx_powers, certainty, gw_id='gateway_under_test') for
            dev in devices}


            # prev_model_1 = pickle.load(open('model_cambridge_5.p', 'rb'))
            # prev_margins_1 = pickle.load(open('margins_cambridge_5.p', 'rb'))
            # self.handlers = {'cambridge_5': Worker_s('cambridge_5', sfs, crs, tx_powers, certainty,
            #                                          gw_id='gateway_under_test', prev_margins=prev_margins_1,
            #                                          prev_model=prev_model_1)
            #                  }

        # finally, register the client and set up callbacks
        ttn_handler = ttn.HandlerClient(self.app_id, self.access_key)
        self.mqtt_client = ttn_handler.data()
        self.mqtt_client.set_uplink_callback(self.meta_handler)
        self.mqtt_client.set_close_callback(self.close_cb)

        self.last_message = dict()
        self.inter_arrivals = dict()
        self.current_estimation = dict()
        for key in self.handlers.keys():
            self.inter_arrivals[key] = list()
            self.current_estimation[key] = list()

    # if the connection is abruptly closed, try to open it again
    def close_cb(self, res, client):
        print("Attempt to close detected. Trying to re-connect")
        client.connect()

    def current_estimation_to_lambda(self):
        """
        Returns the current estimation of lambda for each device
        :return: A dictionary with key = device name and value = lambda (1/s)
        """
        L = len(self.current_estimation)
        lambdas_dict = dict()
        for i in range(1, L + 1):
            lambdas_dict[i] = self.current_estimation['cambridge_{}'.format(i)]

        return lambdas_dict

    def updating_parallel(self):
        """
        Code employed to compute the new optimal C_opt in parallel. Will only trigger an update if the new throughput
        is large enough.
        :return: None
        """
        print("Updating_parallel")
        new_lambdas_dict = self.current_estimation_to_lambda()
        for key, val in new_lambdas_dict.items():
            self.lambdas_[key - 1] = 1 / val

        self.lora.lambdas_ = self.lambdas_
        old_performance = self.lora.compute_network_performance()

        self.lora.find_optimal_c(verbose=False)
        new_config = reduce_to(self.lora.C_opt, 5)
        # self.lora.C = new_config
        # self.lora.C_opt = new_config
        new_performance = -self.lora.get_reward_matricial(new_config)  # self.lora.compute_network_performance()

        assert (new_performance - old_performance) > -0.01

        diff_perf = abs(new_performance - old_performance) / old_performance
        now = time.time()
        diff_time = now - self.last_update

        if diff_perf > 0.005 or (diff_perf > 0.003 and diff_time > 900):
            self.lora.C = new_config
            self.lora.C_opt = new_config
            print(bcolors.WARNING + "Gotta update nodes" + bcolors.ENDC)
            print(new_config)
            self.last_update = now
            self.change_pattern_nodes()

        return

    def change_pattern_nodes(self):
        """
        Function in charge of updating each node's configuration
        :return: None
        """
        for key, val in self.handlers.items():
            if key == 'cambridge_5':
                continue

            node_id = int(key[-1:])
            assert self.lora.C_opt.shape[1] > node_id

            val.update_config(new_config_vector=self.lora.C_opt[1:, node_id - 1])

    def change_pattern_interferer(self):
        """
        Function in charge of making the fifth node increase its lambda value
        :return: None
        """
        print(bcolors.WARNING + "Changing node 5 generation pattern" + bcolors.ENDC)
        new_inter_arrival = 1
        self.handlers['cambridge_5'].update_config(new_rate=new_inter_arrival)

    def re_compute(self):
        """
        Internal function used to compute new C_opt
        :return: None
        """
        self.lora = LoRaWorld(self.lambdas_, self.lengths, self.priorities, self.snrs, self.C_opt, 1, self.model_paths,
                              self.acceptable_tx_powers, forced_nodes=self.forced_nodes,
                              forced_configs=self.forced_configs)
        self.lora.find_optimal_c()
        new_config = reduce_to(self.lora.C_opt, 5)
        self.lora.C = new_config
        self.lora.C_opt = new_config

    def meta_handler(self, msg, client):
        """
        Function called every time a packet is received from the TTN. It enroutes the packet to the correct handler
        :param msg: The MSG received
        :param client: The client object
        :return: None
        """
        if msg.dev_id in self.handlers:
            gws_ids = [g.gtw_id for g in msg.metadata.gateways]
            if 'gateway_under_test' in gws_ids:
                if isinstance(next(iter(self.handlers.values())), Worker_n):
                    now = time.time()
                    if msg.dev_id in self.last_message:
                        lm = self.last_message[msg.dev_id]
                        elapsed = now - lm
                        self.inter_arrivals[msg.dev_id].append(elapsed)

                        sub_select = self.inter_arrivals[msg.dev_id][-20:]
                        weights = [0.99 ** n for n in range(len(sub_select))][::-1]

                        self.current_estimation[msg.dev_id] = np.average(sub_select, weights=weights)
                        print("Current estimation of inter-arrival time of node {} is {} seconds".
                              format(msg.dev_id, self.current_estimation[msg.dev_id]))

                    self.last_message[msg.dev_id] = now

                    # if self.p is not None and self.p.is_alive():
                    #     print("Thread still running. Let's wait for it to finish...")
                    #     # estamos computando una nueva solución, no hacer nada
                    # else:
                    #     if all([len(v) >= 10 for v in self.inter_arrivals.values()]) \
                    #             and (now - self.last_update) > 120:
                    #         assert self.p is None or not self.p.is_alive()
                    #
                    #         self.p = Thread(target=self.updating_parallel)
                    #         self.p.daemon = True
                    #         self.p.start()

                # print("Passing message to registered handler")
                self.handlers[msg.dev_id].uplink_callback(msg, client)

            else:
                print("message not received by my gateway")
        else:
            raise Exception('Node note found', msg.dev_id)

    def run(self):
        """
        Definitively connect to the TTN network and sleep for 4 days
        :return:
        """
        self.mqtt_client.connect()
        time.sleep(60 * 60 * 24 * 4)  # 4 days


if __name__ == '__main__':
    bs = Bootstrapper()
    bs.run()
