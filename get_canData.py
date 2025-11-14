import pandas as pd
from openpilot.tools.lib.logreader import LogReader
from opendbc.can.parser import CANParser
import glob
import os

# === CONFIG ===
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000026--c7b3f58ee5/"
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000027--0019206325/"
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000028--a23c9fe785/"
input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/0000002c--200f216251/"
# no trailer
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/00000025--b269abff9d/"

# with trailer
# input_folder = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/"

dbc_name = "hyundai_kia_generic"

# Target signals (columns)
target_signals = ["LAT_ACCEL", "LONG_ACCEL", "YAW_RATE"]
target_message = "ESP12"

# === Process all .zst files in the folder ===
for input_file in glob.glob(os.path.join(input_folder, "*.zst")):
    output_csv = os.path.splitext(input_file)[0] + "_selected_data.csv"

    # === Load log and parser ===
    lr = LogReader(input_file)
    messages = [(target_message, 0)]
    cp = CANParser(dbc_name, messages, bus=0)

    # === Decode and collect signals in wide format ===
    rows = []
    for msg in lr:
        row = {"timestamp": msg.logMonoTime / 1e9}  # convert ns → seconds

        if msg.which() == "can":
            frames = []
            for i in range(len(msg.can)): # 100 Hz
                f = msg.can[i]
                frames.append((f.address, bytes(f.dat), f.src))

            entry = (msg.logMonoTime, frames)
            cp.update([entry])

            if target_message in cp.vl:
                for sig_name in target_signals:
                    val = cp.vl[target_message].get(sig_name, float('nan'))
                    row[sig_name] = round(val, 5) if val is not None else float('nan')
                retrieved_values = [row.get(key) for key in target_signals]
                if any(retrieved_values):   # remove the all zero message!
                    rows.append(row)

        elif msg.which() == "carState":
            row["vEgo"] = round(msg.carState.vEgo, 5)
            row["yawRate"] = round(msg.carState.yawRate, 5)
            row["steeringAngleDeg"] = round(msg.carState.steeringAngleDeg, 5)
            rows.append(row)

        elif msg.which() == 'liveParameters':
            row["roll"] = round(msg.liveParameters.roll, 5)
            rows.append(row)

        elif msg.which() == 'carControl':
            row["currentCurvature"] = round(msg.carControl.currentCurvature, 5)
            rows.append(row)

        elif msg.which() == 'drivingModelData':
            row["desiredCurvature"] = round(msg.drivingModelData.action.desiredCurvature, 5)
            rows.append(row)


    # === Save to CSV, sorted by timestamp ===
    df = pd.DataFrame(rows)
    df = df.sort_values(by="timestamp")
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved decoded CAN signals to {output_csv}")


# import pandas as pd
# from openpilot.tools.lib.logreader import LogReader
# from opendbc.can.parser import CANParser

# # === CONFIG ===
# input_file = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/1_processed_rlog.zst"
# dbc_name   = "hyundai_kia_generic"
# output_csv = "/home/hatci/openpilot/tools/Downloads/b330cc9641b49158/0000000c--22e39c2f91/1_selected_data.csv"

# # Target signals (columns)
# target_signals = ["LAT_ACCEL", "LONG_ACCEL", "YAW_RATE"]
# target_message = "ESP12"

# # === Load log and parser ===
# lr = LogReader(input_file)
# messages = [(target_message, 0)]
# cp = CANParser(dbc_name, messages, bus=0)

# # === Decode and collect signals in wide format ===
# rows = []
# # previous=0
# for msg in lr:
#     if msg.which() == "can":
#         frames = []
#         for i in range(len(msg.can)): # 100 Hz
#             f = msg.can[i]
#             frames.append((f.address, bytes(f.dat), f.src))

#         entry = (msg.logMonoTime, frames)
#         cp.update([entry])

#         if target_message in cp.vl:
#             row = {"timestamp": round(msg.logMonoTime / 1e9, 3)}  # convert ns → seconds
#             for sig_name in target_signals:
#                 val = cp.vl[target_message].get(sig_name, float('nan'))
#                 row[sig_name] = round(val, 3) if val is not None else float('nan')
#             rows.append(row)

#     if msg.which() == "carState":   # 100 Hz
#         row = {"timestamp": round(msg.logMonoTime / 1e9, 3)}  # convert ns → seconds
#         row["vEgo"] = round(msg.carState.vEgo, 3)
#         row["yawRate"] = round(msg.carState.yawRate, 3)
#         row["steeringAngleDeg"] = round(msg.carState.steeringAngleDeg, 3)
#         rows.append(row)

#     if msg.which() == 'liveParameters':  # 20 Hz
#         row = {"timestamp": round(msg.logMonoTime / 1e9, 3)}  # convert ns → seconds
#         row["roll"] = round(msg.liveParameters.roll,3)
#         rows.append(row)

# # === Save to CSV ===
# df = pd.DataFrame(rows)
# df.to_csv(output_csv, index=False)
# print(f"✅ Saved decoded CAN signals to {output_csv}")
