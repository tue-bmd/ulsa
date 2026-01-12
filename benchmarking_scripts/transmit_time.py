"""Simple script to calculate transmit time for ultrasound imaging."""

depth = 15  # cm
speed_of_sound = 1540  # m/s in soft tissue
n_transmits = 112


transmit_time = (2 * depth / 100) / speed_of_sound  # Convert cm to m
transmit_time_us = transmit_time * 1e6  # Convert seconds to microseconds
print(f"Transmit time for depth {depth} cm: {transmit_time_us:.3f} us")

total_time = transmit_time * n_transmits
total_time_ms = total_time * 1e3  # Convert seconds to milliseconds
print(f"Total transmit time for {n_transmits} transmits: {total_time_ms:.3f} ms")

total_fps = 1 / total_time
print(f"Total frames per second (fps): {total_fps:.2f} fps")
