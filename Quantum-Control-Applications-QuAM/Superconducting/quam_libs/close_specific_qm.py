from quam_libs.components import QuAM
import os
import json
import tomli as tomllib
from pathlib import Path

def get_ports_from_state_json(state_data):
    """
    從 state.json 的 'ports' 鍵值中提取硬體使用狀況
    回傳格式: {(controller, fem): [ports]}
    """
    summary = {}
    ports_container = state_data.get('ports', {})
    
    # 遍歷不同的端口容器，例如 'analog_outputs', 'mw_outputs', 'mw_inputs' 等
    for container_name, controllers in ports_container.items():
        if not isinstance(controllers, dict): continue
        
        for con_id, fems in controllers.items():
            if not isinstance(fems, dict): continue
            
            for fem_id, p_list in fems.items():
                if not isinstance(p_list, dict): continue
                
                # 遍歷該 FEM 下所有的 port 條目
                for port_id_str, port_details in p_list.items():
                    # 確保這是一個有效的端口定義 (包含 controller_id, fem_id, port_id)
                    con = port_details.get('controller_id')
                    fem = port_details.get('fem_id')
                    port = port_details.get('port_id')
                    
                    if con and fem is not None and port is not None:
                        key = (con, fem)
                        if key not in summary:
                            summary[key] = set()
                        summary[key].add(port)
                        
    return {k: sorted(list(v)) for k, v in summary.items()}

def find_matching_config_folders(P, target_config):
    # 1. 取得目標配置 A 的硬體指紋
    # 這裡重複使用之前的邏輯，確保格式一致
    def get_target_summary(config):
        s = {}
        elements = config.get('elements', {})
        for details in elements.values():
            for k in ['MWInput', 'MWOutput', 'singleInput']:
                if k in details and 'port' in details[k]:
                    p_info = details[k]['port']
                    if isinstance(p_info, (tuple, list)) and len(p_info) == 3:
                        con_fem = (p_info[0], p_info[1])
                        if con_fem not in s: s[con_fem] = set()
                        s[con_fem].add(p_info[2])
        return {k: sorted(list(v)) for k, v in s.items()}

    target_fingerprint = get_target_summary(target_config)
    results = []

    # 2. 遍歷資料夾
    for folder_name in os.listdir(P):
        folder_path = os.path.join(P, folder_name)
        state_file = os.path.join(folder_path, 'state.json')
        
        if os.path.isdir(folder_path) and os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取該資料夾的端口指紋
                    current_fingerprint = get_ports_from_state_json(data)
                    
                    # 3. 比對指紋是否完全相同
                    if current_fingerprint == target_fingerprint:
                        results.append(folder_name)
            except Exception as e:
                pass # 忽略讀取錯誤或格式不合的檔案

    return results

def get_hardware_summary(config):
    # 用來暫存資料的字典，結構為: {(controller, fem): set(ports)}
    raw_mapping = {}
    
    elements = config.get('elements', {})
    
    for details in elements.values():
        # 遍歷 element 中可能包含 port 資訊的區塊
        for key in ['MWInput', 'MWOutput', 'singleInput']:
            if key in details and 'port' in details[key]:
                port_info = details[key]['port']
                
                # 確保格式為 ('con1', 1, 5)
                if isinstance(port_info, (tuple, list)) and len(port_info) == 3:
                    con, fem, port = port_info
                    
                    # 使用 (con, fem) 作為 key，並用 set 收集 port 以避免重複
                    if (con, fem) not in raw_mapping:
                        raw_mapping[(con, fem)] = set()
                    raw_mapping[(con, fem)].add(port)
    
    # 將暫存資料轉換成你要求的格式
    result = []
    for (con, fem), ports in raw_mapping.items():
        result.append({
            "controller": con,
            "fems": fem,
            "port": sorted(list(ports))  # 轉成 list 並排序
        })
    
    # 根據 controller 和 fem 排序，讓輸出更整齊
    result.sort(key=lambda x: (x['controller'], x['fems']))
    return result


def close_specific_qm():
    
    machine = QuAM.load()
    qmm = machine.connect()
    qualibrate_config_path = Path.home() / ".qualibrate" / "config.toml"
    quam_state_path = None
    if qualibrate_config_path.exists():
        config = tomllib.loads(qualibrate_config_path.read_text())
        quam_state_path = config.get("quam", {}).get("state_path", None)

    currently_using_qpu = os.path.split(quam_state_path)[-1]
    print("QPU: ",currently_using_qpu)

    qm_ids = list(qmm.list_open_qms())

    if not qm_ids:
        print("No open qms.")
    else:
        for index, qm_id in enumerate(qm_ids):
            print(f"{index}: {qm_id}")

    qm_session_to_close = None

    if len(qm_ids) > 0:
        for index, qm_id in enumerate(qm_ids):

            selected_qm_id = qm_ids[index]
            qm = qmm.get_qm(selected_qm_id)

            #Inspect the selected qm config to verify it's the one blocking the resource you want to release
            config_dict = qm.get_config()
            # 執行並列印
            summary = get_hardware_summary(config_dict)
            print("Hardwares now are runnging:")
            for item in summary:
                print(item)
            
            # current_file_path = os.getcwd()
            # current_dir = os.path.dirname(current_file_path)
            # folder_abs_path = os.path.join(current_dir, "configuration", "quam_state")

            path = Path("~/.qualibrate").expanduser()
            toml_path = path.joinpath("config.toml")
            folder_abs_path = None
            with open(toml_path, "rb") as f:
                folder_abs_path = os.path.join(os.path.split(tomllib.load(f)['qualibrate']['calibration_library']['folder'])[0], "configuration", "quam_state")


            matched_folders = find_matching_config_folders(folder_abs_path, config_dict)
            for folder_name in matched_folders:
                if folder_name == currently_using_qpu:
                    print(f"We see resources overlapping with {folder_name} !")
                    qm_session_to_close = index
    
        if qm_session_to_close is None:
            qm = None
            print(f"After checking deeply, We did NOT see any resources overlapping to your QPU: {currently_using_qpu} !")
        else:
            qm = qmm.get_qm(qm_ids[qm_session_to_close])
            print(f"Detected session index: {qm_session_to_close}, qm-id: {qm_ids[qm_session_to_close]} needs to be closed !")

    else:
        print("No qm session now is running !")
        qm = None


    if qm is not None:
        qm.close()
        print(f"Closed qm index: {qm_session_to_close}, qm-id: {qm_ids[qm_session_to_close]}")
        
    else:
        print("No qm session to close.")


if __name__ == "__main__":
    close_specific_qm()