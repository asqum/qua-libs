import csv
import json
import os

class json_to_csv:
    def __init__(self, state_folder_name:str):
        self.state_folder = state_folder_name
        self.dataset = []
        self.output_csv_name = ""
        self._search_state_wiring()

    def _search_state_wiring(self):
        # 獲取絕對路徑
        current_file_path = os.path.abspath(__file__)

        # 獲取資料夾路徑
        current_dir = os.path.dirname(current_file_path)

        folder_abs_path = os.path.join(current_dir, "quam_state", self.state_folder)
        # state.json data
        self.statefile = open(os.path.join(folder_abs_path, 'state.json'), "r")
        self.state_data = json.load(self.statefile)
        # wiring.json data
        self.wiringfile = open(os.path.join(folder_abs_path, 'wiring.json'),"r")
        self.wiring_data = json.load(self.wiringfile)

        # output csv name definition
        self.output_csv_name = f"{self.state_folder}_QPUinfo.csv"



    def basic_information(self):

        #寫入標題
        self.qubit_id = [""]
        for i in self.state_data["qubits"]:
            self.qubit_id.append(i)
        self.dataset.append(self.qubit_id)
        # 寫入資料
        # resonator frequency
        rf = ["fr (GHz)"]
        for id in self.qubit_id[1:]:
            slot = self.wiring_data["wiring"]["qubits"][id]["rr"]["opx_output"][-3]
            channel = self.wiring_data["wiring"]["qubits"][id]["rr"]["opx_output"][-1]
            IF = self.state_data["qubits"][id]["resonator"]["intermediate_frequency"]
            LO = self.state_data["ports"]["mw_outputs"]["con1"][slot][channel]["upconverter_frequency"]
            rf.append(round((LO+IF)/1e9,3))
        self.dataset.append(rf)
        # qubit frequency
        qf = ['fq (GHz)']
        for id in self.qubit_id[1:]:
            slot = self.wiring_data["wiring"]["qubits"][id]["xy"]["opx_output"][-3]
            channel = self.wiring_data["wiring"]["qubits"][id]["xy"]["opx_output"][-1]
            IF = self.state_data["qubits"][id]["xy"]["intermediate_frequency"]
            LO = self.state_data["ports"]["mw_outputs"]["con1"][slot][channel]["upconverter_frequency"]
            if IF == 0:
                qf.append("")
            else:
                qf.append(round((LO+IF)/1e9,3))
        self.dataset.append(qf)
        # T1
        T1_list = ['T1 (us)']
        for id in self.qubit_id[1:]:
            if "T1" not in self.state_data["qubits"][id]:
                T1 = ""
            else:
                T1 = self.state_data["qubits"][id]["T1"]
                T1 = round(T1*1e6,1)
            T1_list.append(T1)
        self.dataset.append(T1_list)
        # T2*
        T2ramsey_list = ['T2* (us)']
        for id in self.qubit_id[1:]:
            if "T2ramsey" not in self.state_data["qubits"][id]:
                T2ramsey = ""
            else:
                T2ramsey = self.state_data["qubits"][id]["T2ramsey"]
                T2ramsey = round(T2ramsey*1e6,1)
            T2ramsey_list.append(T2ramsey)
        self.dataset.append(T2ramsey_list)
        # T2
        T2echo_list = ['T2 (us)']
        for id in self.qubit_id[1:]:
            if "T2echo" not in self.state_data["qubits"][id]:
                T2 = ""
            else:
                T2 = self.state_data["qubits"][id]["T2echo"]
                T2 = round(T2*1e6,1)
            T2echo_list.append(T2)
        self.dataset.append(T2echo_list)
        # flux tunable
        tunable = ["tunable"]
        for id in self.qubit_id[1:]:
            if "independent_offset" not in self.state_data["qubits"][id]["z"]:
                tunable.append("X")
            else:
                tunable.append("O")
        self.dataset.append(tunable)

    def add_information(self,additional_information):
        name, title = additional_information[0], additional_information[1]
        list = [f"{title}"]
        for id in self.qubit_id[1:]:
            if f"{name}" not in self.state_data["qubits"][id]:
                list.append("")
            else:
                list.append(self.state_data["qubits"][id][name])
        self.dataset.append(list)
    def write_information(self):
        with open(self.output_csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.dataset)

if __name__ == "__main__":
    ########################## Parameters ##########################
    # state and wiring parent directory name
    parent_dir_name = "as-qpu-10qV2_q1q5"
    #extra information need to add into form
    additional_information = [] # fill in the format: [name, title], name: variable in json, ex: ["T1","T1 (us)"]
    ################################################################

    # execution
    j_to_c = json_to_csv(parent_dir_name)
    j_to_c.basic_information()
    if additional_information:
        j_to_c.add_information(additional_information)
    j_to_c.write_information()
