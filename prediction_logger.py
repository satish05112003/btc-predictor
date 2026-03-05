import csv
import os
from datetime import datetime

FILE="last_predictions.csv"

def log_prediction(tf,signal,price,p_up,p_down,confidence):

    file_exists=os.path.exists(FILE)

    with open(FILE,"a",newline="") as f:

        writer=csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "tf",
                "signal",
                "price",
                "p_up",
                "p_down",
                "confidence"
            ])

        writer.writerow([
            datetime.utcnow(),
            tf,
            signal,
            price,
            p_up,
            p_down,
            confidence
        ])
