"""Convert the public DCS dashboard's weekly data script to a wide CSV."""

import argparse
import calendar
import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path

SOURCE = "https://www.statistics.gov.lk/DashBoard/Prices/Prices_Data.php"
ROOT = Path(__file__).resolve().parents[1]
MONTHS = {name: number for number, name in enumerate(calendar.month_abbr) if name}


def read_array(source, name):
    match = re.search(rf"(?ms)^var {name}\s*=\s*(\[.*?\])\s*;", source)
    if match is None:
        raise ValueError(f"DCS array {name!r} is missing")
    # The script uses an empty single-quoted string to mark unavailable prices.
    return json.loads(re.sub(r":\s*''", ": null", match[1]))


def period_date(label):
    week, month, year = label.split(".")
    # The DCS dashboard publishes week-of-month labels, with four per month.
    # Days 7, 14, 21 and 28 are period markers, independent of publication day.
    return date(int(year), MONTHS[month[:3]], 7 * int(week.removeprefix("W")))


def import_dcs(script_path, output_path):
    raw = script_path.read_bytes()
    source = raw.decode("utf-8-sig")
    products = read_array(source, "pip")
    reports = read_array(source, "prices")
    columns = [" ".join(product["name"].split()) for product in products]
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate product names in dashboard source")
    dated_reports = sorted((period_date(row["Date"]), row) for row in reports)
    if len(dated_reports) != len({day for day, _ in dated_reports}):
        raise ValueError("Duplicate weekly periods in dashboard source")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        file.write(f"# Source: Sri Lanka Department of Census and Statistics, {SOURCE}\n")
        file.write(f"# Source script SHA-256: {hashlib.sha256(raw).hexdigest()}\n")
        file.write("# Colombo District open-market weekly average retail prices (LKR).\n")
        file.write("# period: DCS week-of-month label mapped to day 7, 14, 21 or 28.\n")
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["period", *columns])
        for day, report in dated_reports:
            writer.writerow([day, *(report.get(item["product"]) for item in products)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="Saved Prices_Data.php response")
    parser.add_argument("--output", type=Path, default=ROOT / "data/retail-weekly-w.csv")
    args = parser.parse_args()
    import_dcs(args.script, args.output)
