import json, os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', ".."))
from Superconducting.calibration_graph.RatisRetuneV1 import g
import time, threading, logging
from datetime import datetime

# 設定 logging：輸出到 log.txt，並且也印在螢幕上
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

target_q = ["q1", "q2", "q3", "q4", "q5"]
clock_at = ["09:30", "13:30", "17:30", "21:30", "01:30", "05:30"]

already_ran = set()  

def task(target_time):
    logging.info(f"start calibration at {target_time}")
    g.run(qubits=target_q)
    logging.info(f"Calibration completed at {target_time}")

while True:
    now = datetime.now()
    today = now.date()

    for t in clock_at:
        target = datetime.strptime(t, "%H:%M").replace(
            year=today.year, month=today.month, day=today.day
        )
        delta = abs((now - target).total_seconds()) / 60  # 分鐘差

        if delta <= 5 and (today, t) not in already_ran:
            print(f"Doing mission at {t}")
            threading.Thread(target=task, args=(t,)).start()
            already_ran.add((today, t))

    if len(clock_at) == len(list(already_ran)):
        break
    time.sleep(30)
    
    

