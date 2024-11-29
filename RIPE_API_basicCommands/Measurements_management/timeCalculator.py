# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:57:40 2023

@author: antonio
"""

import tkinter as tk

def create_ui():

    root = tk.Tk()
    root.title("Compute duration of measurement")
    result_label = tk.Label(root, text="")

    def getInt(ee):
        input_val = ee.get()
        if input_val == '':
            input_val = 1

        return float(input_val)

    def measurementCost(*args):
        unit_cost = 10*getInt(e3)*(int(getInt(e4)/1500) + 1)

        times_perDay = 24*60 / getInt(e5)
        daily_cost = unit_cost*getInt(e1)*getInt(e2)*times_perDay

        if daily_cost - getInt(e7) <= 0:
            total_time = float('inf')
        else:
            total_time = getInt(e6)/(daily_cost - getInt(e7))

        print(total_time)

        result_label.config(text=f" {total_time:.2f}")




    # Create three labels
    tk.Label(root, text="     ").grid(row=0)
    tk.Label(root, text="Num. sources").grid(row=1)
    tk.Label(root, text="Num. destinations").grid(row=2)

    tk.Label(root, text="      ").grid(row=3)

    tk.Label(root, text="Num. packets").grid(row=4)
    tk.Label(root, text="Size packets (Bytes)").grid(row=5)

    tk.Label(root, text="      ").grid(row=6)

    tk.Label(root, text="Frequency (mins)").grid(row=7)

    tk.Label(root, text="Starting credits").grid(row=8)
    tk.Label(root, text="Daily earned credits").grid(row=9)

    tk.Label(root, text="      ").grid(row=10)

    # Create three entry fields
    e1 = tk.Entry(root)
    e2 = tk.Entry(root)
    e3 = tk.Entry(root)
    e4 = tk.Entry(root)
    e5 = tk.Entry(root)
    e6 = tk.Entry(root)
    e7 = tk.Entry(root)

    field_list = [e1,e2,e3,e4,e5,e6,e7]

    # Grid layout manager to put the entry fields in correct locations
    e1.grid(row=1, column=1)
    e2.grid(row=2, column=1)

    e3.grid(row=4, column=1)
    e4.grid(row=5, column=1)

    e5.grid(row=7, column=1)
    e6.grid(row=8, column=1)
    e7.grid(row=9, column=1)

    num_packets = 3 # num packets per time
    size_packet = 1499

    num_src = 40
    num_dst = 8

    time_delta_min  = 30
    cred_start      = int(18983320) # at the end of 2023-11-19
    cred_earned_day = 125000


    e1.insert(0, num_src)
    e2.insert(0, num_dst)
    e3.insert(0, num_packets)
    e4.insert(0, size_packet)
    e5.insert(0, time_delta_min)
    e6.insert(0, cred_start)
    e7.insert(0, cred_earned_day)

    # Create a result label
    tk.Label(root, text="Minimum number of days:").grid(row=11)
    measurementCost()
    result_label.grid(row=11, column=1)


    # Update the result label whenever the entry fields change
    for efield in field_list:
        efield.bind("<KeyRelease>", measurementCost)


    # Start the event loop
    root.mainloop()

if __name__ == "__main__":
    create_ui()
