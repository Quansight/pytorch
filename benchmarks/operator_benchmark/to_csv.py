"""

Usage:
$ python -m pt.interpolate_test --ai_pep_format --tag_filter=all > bench_interpolate.log
$ python to_csv.py bench_interpolate.log bench_interpolate.csv

"""
from pathlib import Path
import argparse
import csv
import json


def convert(input_filepath, output_filepath):
    c = 0
    with Path(input_filepath).open("r") as h:
        with Path(output_filepath).open("w") as w:

            csvwriter = csv.writer(w, delimiter=',')            
            header = None

            while True:
                current_line = h.readline()
                if len(current_line) < 1:
                    break

                if not current_line.startswith("PyTorchObserver"):
                    continue

                current_line = current_line[len("PyTorchObserver "):]
                data = json.loads(current_line)
                assert "config" in data, "config key should be in data, got {}".format(list(data.keys()))
                # parse input config as a dict
                config_str = data["config"]
                config = [c.split(": ") for c in config_str.split(", ")]
                assert all([len(c) == 2 for c in config]), "Bad parsed config: {}".format(config)
                del data["config"]
                data.update({c[0]: c[1] for c in config})

                if header is None:
                    header = sorted(data.keys())
                    csvwriter.writerow(header)

                csvwriter.writerow([data.get(h, "") for h in header])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert AI-PEP logs to CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input_filepath',
        help='Input logs filepath in AI-PEP format',
    )
    parser.add_argument('output_filepath',
        help='Output CSV filepath',
    )
    args, _ = parser.parse_known_args()

    if not Path(args.input_filepath).exists():
        raise ValueError("Input file '{}' is not found".format(args.input_filepath))

    if Path(args.output_filepath).exists():
        raise ValueError("Existing output file '{}'. Please remove it before running this script".format(args.output_filepath))

    convert(args.input_filepath, args.output_filepath)
