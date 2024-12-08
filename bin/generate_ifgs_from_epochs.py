#!/usr/bin/env python3

from datetime import timedelta, datetime

# Function to generate pairs up to a specified length
def generate_ifg_pairs(start_date, end_date, interval, lengths):
    epochs = [
        (start_date + timedelta(n)).strftime("%Y%m%d")
        for n in range(0, (end_date - start_date).days + 1, interval)
    ]

    all_ifgs = []
    for length in lengths:
        ifgs = [
            f"{epochs[i]}_{min((start_date + timedelta(n + length)).strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))}"
            for i, n in enumerate(range(0, (end_date - start_date).days + 1, interval))
            if n + length <= (end_date - start_date).days
        ]
        all_ifgs.extend(ifgs)

    return all_ifgs

