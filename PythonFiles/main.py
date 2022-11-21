import gym_donkeycar
import gym
import cv2
import numpy as np
import pygame
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean

pygame.init()
# sterowanie = axis 2;  hamulec = axis 4;  gaz = axis 5

env = gym.make("donkey-generated-track-v0")

obs = env.reset()

def process_image(img):
    processed = cv2.resize(img, (800,600))
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    processed = cv2.Canny(processed, threshold1=50, threshold2=100)
    processed = cv2.GaussianBlur(processed,(11,11),0)
    
    mask = np.zeros_like(processed)
    vertices = np.array([[10,400],[10,250],[300,200],[500,200],[800,250],[800,400],], np.int32)
    cv2.fillPoly(mask, [vertices], 255)
    processed = cv2.bitwise_and(processed, mask)

    return processed


def end_lanes(h_lines):
    try:
        # y do rysowania linii
        ys = []
        for i in h_lines:
            for j in i:
                ys += [j[1], j[3]] #wszytkie y
            min_y = min(ys)
            max_y = 400

            line_dict = {}          
        #return min_y
        # ogarniecie funkcji linii do wyznaczania najwazniejszych
        for idx, i in enumerate(h_lines):
            for xyxy in i:
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
                try:
                    A = vstack([x_coords, ones(len(x_coords))]).T
                    a, b = lstsq(A, y_coords, rcond=None)[0]
                    if a == 0.0: pass
                    else:
                        x1 = (min_y-b) / a
                        x2 = (max_y-b) / a
            
                        line_dict[idx] = [a,b,[int(x1), min_y, int(x2), max_y]]
                except: 
                    pass


        final_lanes = {}
        threshold = 0.1

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]


        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except: pass



def lines(img):

    lines = cv2.HoughLinesP(img, 1, np.pi/180, 180, np.array([]), 100, 5)
    #draw_lines(img, lines)
    
    return lines

def draw_lines(img, lines):
    try:
        for line, val in enumerate(lines):
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
    except: pass

while True:
    for event in pygame.event.get(): # User did something

        joystick = pygame.joystick.Joystick(0)
        joystick.init()
         
    p_img = process_image(obs)
    final_lanes_coords = lines(p_img)
    print(final_lanes_coords)
    if end_lanes(final_lanes_coords) != None:
        line1, line2 = end_lanes(final_lanes_coords)
        try:
            cv2.line(p_img, (int(line1[0]), int(line1[1])), (int(line1[2]), int(line1[3])), (100,255,0), 20)
            cv2.line(p_img, (int(line2[0]), int(line2[1])), (int(line2[2]), int(line2[3])), (100,0,255), 20)
        except:
            pass
    else:   
        pass

    cv2.imshow("test2", p_img)

    str = round(joystick.get_axis(2), 1)
    brk = (joystick.get_axis(4) + 1) / 2
    acc = (joystick.get_axis(5) + 1) / 2
    fwrd = round(acc - brk, 1)

    controls = [str, fwrd]

    obs, reward, done, infos = env.step(controls)
    #print(controls)

    cv2.waitKey(0)