import csv
import json
import os

class json_to_csv:
    def __init__(self, state_folder_name: str):
        self.state_folder = state_folder_name
        self.dataset = []
        self.output_csv_name = ""
        # 定義符號：尚未測量
        self.NOT_MEASURED = "---" 
        self._search_state_wiring()

    def _search_state_wiring(self):
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        folder_abs_path = os.path.join(current_dir, "quam_state", self.state_folder)
        
        # 讀取 JSON 檔案
        with open(os.path.join(folder_abs_path, 'state.json'), "r", encoding="utf-8") as f:
            self.state_data = json.load(f)
        with open(os.path.join(folder_abs_path, 'wiring.json'), "r", encoding="utf-8") as f:
            self.wiring_data = json.load(f)

        self.output_csv_name = f"{self.state_folder}_QPUinfo.csv"

    def basic_information(self):
        # 1. 寫入標題列 (Qubit IDs)
        self.qubit_id = ["Qubit ID"]
        for i in self.state_data["qubits"]:
            self.qubit_id.append(i)
        self.dataset.append(self.qubit_id)

        # 2. Resonator frequency (fr)
        rf = ["fr (GHz)"]
        for id in self.qubit_id[1:]:
            try:
                slot = self.wiring_data["wiring"]["qubits"][id]["rr"]["opx_output"][-3]
                channel = self.wiring_data["wiring"]["qubits"][id]["rr"]["opx_output"][-1]
                IF = self.state_data["qubits"][id]["resonator"]["intermediate_frequency"]
                LO = self.state_data["ports"]["mw_outputs"]["con1"][slot][channel]["upconverter_frequency"]
                rf.append(round((LO + IF) / 1e9, 3))
            except:
                rf.append(self.NOT_MEASURED)
        self.dataset.append(rf)

        # 3. Qubit frequency (fq)
        qf = ['fq (GHz)']
        for id in self.qubit_id[1:]:
            try:
                IF = self.state_data["qubits"][id]["xy"]["intermediate_frequency"]
                if IF == 0:
                    qf.append(self.NOT_MEASURED)
                else:
                    slot = self.wiring_data["wiring"]["qubits"][id]["xy"]["opx_output"][-3]
                    channel = self.wiring_data["wiring"]["qubits"][id]["xy"]["opx_output"][-1]
                    LO = self.state_data["ports"]["mw_outputs"]["con1"][slot][channel]["upconverter_frequency"]
                    qf.append(round((LO + IF) / 1e9, 3))
            except:
                qf.append(self.NOT_MEASURED)
        self.dataset.append(qf)

        # 4. T1, T2*, T2 (批次處理相似邏輯)
        metrics = [
            ("T1", "T1 (us)", 1e6),
            ("T2ramsey", "T2* (us)", 1e6),
            ("T2", "T2 (us)", 1e6)
        ]
        for key, title, scale in metrics:
            row = [title]
            for id in self.qubit_id[1:]:
                val = self.NOT_MEASURED
                extras = self.state_data["qubits"][id].get("extras", {})
                if key in extras:
                    main_val = extras[key]
                    dev_key = f"{key}_dev"
                    if dev_key in extras:
                        val = f"{round(main_val*scale, 1)} \u00B1 {round(extras[dev_key]*scale, 2)}"
                    else:
                        val = str(round(main_val * scale, 1))
                row.append(val)
            self.dataset.append(row)

        # 5. Readout fidelity
        RO = ["RO_fidelity"]
        for id in self.qubit_id[1:]:
            res = self.state_data["qubits"][id].get("resonator", {})
            if 'confusion_matrix' in res:
                p00 = float(res["confusion_matrix"][0][0])
                p11 = float(res["confusion_matrix"][1][1])
                RO.append(str(round(0.5 * (p00 + p11), 3)))
            else:
                RO.append(self.NOT_MEASURED)
        self.dataset.append(RO)

        # 6. Effective temperature (Teff)
        Teff = ['Teff (mK)']
        for id in self.qubit_id[1:]:
            val = self.NOT_MEASURED
            extras = self.state_data["qubits"][id].get("extras", {})
            if "Teff_mK" in extras:
                t = extras["Teff_mK"]
                dev = extras.get("Teff_mK_dev")
                if dev:
                    val = f"{round(t, 1)} \u00B1 {round(dev, 1)}"
                else:
                    val = f"{round(t, 1)}"
            Teff.append(val)
        self.dataset.append(Teff)

        # 7. Flux tunable (保持原本的 O/X 邏輯)
        tunable = ["tunable"]
        for id in self.qubit_id[1:]:
            val = self.NOT_MEASURED
            period_V = float(self.state_data["qubits"][id]['phi0_voltage'])
            if period_V > 0 and period_V<=1.5:
                tunable.append("O")
            else:
                tunable.append("X")
        self.dataset.append(tunable)

    def add_information(self, additional_info):
        # 處理額外資訊，找不到則填 ---
        name, title = additional_info[0], additional_info[1]
        new_row = [title]
        for id in self.qubit_id[1:]:
            val = self.state_data["qubits"][id].get(name, self.NOT_MEASURED)
            new_row.append(val)
        self.dataset.append(new_row)

    def write_information(self):
        
        with open(self.output_csv_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(self.dataset)

        print(f"--- 報告產生完成 ---")
        print(f"檔案路徑: {os.path.abspath(self.output_csv_name)}")

if __name__ == "__main__":
    # 設定資料夾名稱
    parent_dir_name = "as-qpu-10qV2"
    
    # 初始化並執行
    j_to_c = json_to_csv(parent_dir_name)
    j_to_c.basic_information()
    
    # 執行寫入
    j_to_c.write_information()