"""
Run this file to test the trained model on pre-recorded videos.
"""



from ultralytics import YOLO
import cv2

model = YOLO("/home/kalgaonp/runs/detect/runs/train/drone_rpi5-5/weights/best.pt")

video_path = "test_videos/CINE CHASE ¦ FPV Drone Chasing.mp4"
cap = cv2.VideoCapture(video_path)

#get video properties for the output writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

#Set up output video writer — avc1 (H.264) produces broadly compatible mp4 files;
# fall back to XVID in .avi if avc1 is not available in this OpenCV build
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
if not out.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

#Process frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # end of video

    #Run YOLO interface on the frame
    results = model.predict(source=frame, conf=0.25, verbose=False)

    #Draw bounding boxes on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls]} {conf:.2f}"
            
            #Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    #Save the annotated frame
    out.write(frame)

#Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Done. Annotated video saved to output.mp4")