import csv
import os
from datetime import datetime

FILE = "prediction_performance.csv"

def log_result(tf,prediction,actual):

    correct = prediction == actual

    with open(FILE,"a",newline="") as f:
        writer=csv.writer(f)
        
        # Write header if file is empty
        if not os.path.exists(FILE) or os.path.getsize(FILE) == 0:
            writer.writerow(["timestamp", "timeframe", "prediction", "actual_direction", "correct"])
            
        writer.writerow([
            datetime.utcnow(),
            tf,
            prediction,
            actual,
            correct
        ])

def get_stats():

    total=0
    correct=0

    if not os.path.exists(FILE):
        return 0,0,0

    with open(FILE) as f:
        reader=csv.DictReader(f)

        for r in reader:
            total+=1
            if r["correct"]=="True":
                correct+=1

    accuracy = (correct/total*100) if total else 0

    return total,correct,accuracy

def get_today_stats():

    today=datetime.utcnow().date()

    total=0
    correct=0
    
    if not os.path.exists(FILE):
        return 0,0

    with open(FILE) as f:
        reader=csv.DictReader(f)

        for r in reader:

            t=datetime.fromisoformat(r["timestamp"])

            if t.date()==today:
                total+=1
                if r["correct"]=="True":
                    correct+=1

    accuracy=(correct/total*100) if total else 0

    return total,accuracy

def get_mistakes():

    mistakes=0
    
    if not os.path.exists(FILE):
        return 0

    with open(FILE) as f:
        reader=csv.DictReader(f)

        for r in reader:
            if r["correct"]=="False":
                mistakes+=1

    return mistakes
