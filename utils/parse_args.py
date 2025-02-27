import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse arguments for FitAgent')
    parser.add_argument('--type', required=True, help='Type of the measurement')
    parser.add_argument('--value', required=True, type=int, help='Value of the measurement')
    parser.add_argument('--unit', required=True, help='Unit of the measurement')
    parser.add_argument('--startdate', required=True, help='Start date of the measurement')
    parser.add_argument('--enddate', help='End date of the measurement')
    return parser.parse_args()