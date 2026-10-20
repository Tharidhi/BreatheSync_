"""Download the labelled Apnea-ECG learning set from PhysioNet (~200 MB).

Usage:  python src/download_data.py            # all 35 labelled records
        python src/download_data.py a01 a02    # a subset (quick test)
"""
import sys
import wfdb
from config import RAW_DIR, LEARNING_RECORDS


def main():
    records = sys.argv[1:] or LEARNING_RECORDS
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(records)} records to {RAW_DIR} ...")
    wfdb.dl_database("apnea-ecg", str(RAW_DIR), records=records, annotators=["apn"])
    print("Done.")


if __name__ == "__main__":
    main()
