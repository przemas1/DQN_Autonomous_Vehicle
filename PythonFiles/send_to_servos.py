import serial
import time
print(1)
ser = serial.Serial(port='COM3', baudrate=115200, bytesize=8, timeout=3, stopbits=serial.STOPBITS_ONE)
while True:

    ser.write(bytearray([255, 255]))
    time.sleep(1)
    ser.write(b'\x00\x00')
    time.sleep(1)

