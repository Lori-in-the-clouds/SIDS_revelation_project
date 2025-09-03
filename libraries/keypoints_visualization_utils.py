import cv2

def draw_keypoints(img_path, label_kpt_path,output_path,number:bool=False,thickness_line=1,thickness_point=3):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    skeleton = [(0, 6),(0, 5),(6,8),(0,1),(0,2),(6,5),(6,4),(5,3),(4,2),(3,1),(6,12),(5,7),(5,11),(7,9),(8,10),(12,11),(12,14),(14,16),(11,13),(13,15)]

    with open(label_kpt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            kp_data = parts[5:]
            keypoints = []
            for idx in range(0, len(kp_data), 3):
                x_norm = float(kp_data[idx])
                y_norm = float(kp_data[idx+1])
                v = int(float(kp_data[idx+2]))
                if v > 0:
                    x_px = int(x_norm * w)
                    y_px = int(y_norm * h)
                    keypoints.append((x_px, y_px))
                    # Draw point
                    cv2.circle(img, (x_px, y_px), thickness_point, (0, 0, 255), -1)
                    # Number near to point
                    if number:
                        cv2.putText(img, str(idx//3), (x_px + 3, y_px - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    keypoints.append(None)

            # disegna le linee dello scheletro
            for start, end in skeleton:
                if keypoints[start] is not None and keypoints[end] is not None:
                    if (start == 6 and end == 5) or (start == 6 and end == 12) or (start == 5 and end == 11) or (start == 12 and end == 11):
                        cv2.line(img, keypoints[start], keypoints[end], (255, 0, 255), thickness_line)
                    if (start == 12 and end == 14) or (start == 11 and end == 13) or (start == 14 and end == 16) or (start == 13 and end == 15):
                        cv2.line(img, keypoints[start], keypoints[end], (0, 0, 255), thickness_line)
                    if (start == 6 and end == 8) or (start == 8 and end == 10) or (start == 5 and end == 7) or (start == 7 and end == 9):
                        cv2.line(img, keypoints[start], keypoints[end], (0, 255, 0), thickness_line)
                    if (start == 6 and end == 4) or (start == 4 and end == 2) or (start == 5 and end == 3) or (start == 3 and end == 1) or (start == 0 and end == 2) or (start == 0 and end == 1):
                        cv2.line(img, keypoints[start], keypoints[end], (0, 255, 255), thickness_line)

        # salva immagine finale
        cv2.imwrite(output_path, img)