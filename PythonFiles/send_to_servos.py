import serial
import keyboard
import time

ser = serial.Serial(port='COM4', baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
ser.write(bytes([3]))
ser.write(bytes([2]))
