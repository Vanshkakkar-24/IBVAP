import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

while True:

    success, frame = camera.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        verbose=False
    )

    for result in results:

        if result.boxes.id is None:
            continue

        boxes = result.boxes

        track_ids = boxes.id.int().cpu().tolist()

        coordinates = boxes.xyxy.int().cpu().tolist()

        confidences = boxes.conf.cpu().tolist()

        for track_id, box, confidence in zip(
            track_ids,
            coordinates,
            confidences
        ):

            x1, y1, x2, y2 = box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person #{track_id} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Person Tracking Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()