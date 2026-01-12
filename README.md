# Server Farm Simulation

## Overview

This project simulates a heterogeneous server farm to evaluate the performance of two load balancing algorithms.

Modern server farms often consist of a mix of newer and older servers with different processing capabilities. This simulation models a farm with:

- 3 servers (2 standard-speed, 1 fast-speed)
- A dispatcher that routes jobs based on server queue lengths and a prioritisation parameter

The system processes jobs using first-come-first-served (FCFS) scheduling within each server.

Originally submitted as the major project for COMP9334 Capacity Planning of Computer Systems and Networks (UNSW, T1/2021).

## Directory Structure

```text
server-farm-simulation/
├── README.md
├── run_test.sh                # Shell script to run a test
├── report.pdf
├── src/                       # Python simulation code
│   └── simulate.py            # Main simulation script
├── config/                    # Input configuration files
│   ├── mode_\*.txt
│   ├── para_\*.txt
│   ├── interarrival_\*.txt
│   └── servicetime_\*.txt
├── output/                    # Output files
│   ├── mrt_\*.txt
│   ├── depart1_\*.txt
│   ├── depart2_\*.txt
│   └── depart3_\*.txt
└── requirements.txt
```

## Configuration Files

mode_*.txt: either random or trace

para_*.txt: algorithm parameters

- If random: 4 lines → f, algorithm_version, d, time_end
- If trace: 3 lines → f, algorithm_version, d

interarrival_*.txt:

- If random: λ, a2_lower, a2_upper
- If trace: one interarrival time per line

servicetime_*.txt: required only in trace mode (one service time per line)

## How to Run

Ensure you are in a Linux environment.

chmod +x run_test.sh
./run_test.sh

Note by default, run_test.sh runs test case 1. To run a different test case (e.g. test 2), open run_test.sh and modify:

TEST_ID=2

Your script will read from config/ and write to output/. The results will consist of:

- mrt_*.txt: Mean response time
- depart1_*.txt, depart2_*.txt, depart3_*.txt: Departure times per server

The output folder is intentionally left empty in the repository, except for a placeholder file to preserve the folder structure.

## Report

For details on the system design, assumptions, and performance analysis, see report.pdf.
