import csv
import os
from datetime import datetime

FILE="mistakes_dataset.csv"

def log_mistake(features):

    file_exists=os.path.exists(FILE)

    with open(FILE,"a",newline="") as f:

        writer=csv.writer(f)

        if not file_exists:
            writer.writerow([
            "timestamp",
            "timeframe",
            "prediction",
            "actual_direction",
            "RSI",
            "volume",
            "orderflow_buy",
            "orderflow_sell",
            "imbalance",
            "pressure",
            "volatility"
            ])

        writer.writerow([
            datetime.utcnow(),
            features["tf"],
            features["prediction"],
            features["actual"],
            features["RSI"],
            features["volume"],
            features["buy"],
            features["sell"],
            features["imbalance"],
            features["pressure"],
            features["volatility"]
        ])
