import cv2
import numpy as np
from numpy import ones, vstack
from numpy.linalg import lstsq
from statistics import mean


# zwraca wycięty obraz po algorytmie cannyego
def process_image(img):
    mask = np.zeros_like(img)
    vertices = np.array([[2, 80], [2, 50], [60, 50], [100, 50], [160, 60], [160, 80], ], np.int32)
    processed = cv2.fillPoly(mask, [vertices], 255)
    processed = cv2.bitwise_and(processed, mask)

    # processed = cv2.resize(processed, (800, 600))
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    processed = cv2.Canny(processed, threshold1=50, threshold2=100)
    processed = cv2.GaussianBlur(processed, (11, 11), 0)
    return processed


def end_lanes(h_lines):
    try:
        # y do rysowania linii
        ys = []
        for i in h_lines:
            for j in i:
                # wszystkie y
                ys += [j[1], j[3]]
            min_y = min(ys)
            max_y = 400

            line_dict = {}          

        for idx, i in enumerate(h_lines):
            for xyxy in i:
                x_coords = (xyxy[0], xyxy[2])
                y_coords = (xyxy[1], xyxy[3])
                try:
                    A = vstack([x_coords, ones(len(x_coords))]).T
                    a, b = lstsq(A, y_coords, rcond=None)[0]
                    if a == 0.0:
                        pass
                    else:
                        x1 = (min_y-b) / a
                        x2 = (max_y-b) / a
            
                        line_dict[idx] = [a, b, [int(x1), min_y, int(x2), max_y]]
                except: 
                    pass

        final_lanes = {}
        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]

            if len(final_lanes) == 0:
                final_lanes[m] = [[m, b, line]]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > \
                                    abs(final_lanes_copy[other_ms][0][1] * 0.8):
                                final_lanes[other_ms].append([m, b, line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [[m, b, line]]

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
    except:
        pass


# zwraca koordynaty wykrytych linii
def lines(img):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 180, np.array([]), 100, 5)
    return lines


def draw_lines(img, lines):
    try:
        for line, val in enumerate(lines):
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
    except: pass


def preprocessing(img):

    p_img = process_image(img)
    final_lanes_coords = lines(p_img)

    if end_lanes(final_lanes_coords) is not None:
        line1, line2 = end_lanes(final_lanes_coords)
        try:
            cv2.line(p_img, (int(line1[0]), int(line1[1])), (int(line1[2]), int(line1[3])), (100, 255, 0), 20)
            cv2.line(p_img, (int(line2[0]), int(line2[1])), (int(line2[2]), int(line2[3])), (100, 0, 255), 20)
        except:
            pass
    else:
        pass

    input_image = cv2.resize(p_img, (84, 84))
    cv2.imwrite('test.jpg', input_image)
    return input_image
