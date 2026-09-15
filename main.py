import cv2
import numpy as np
import json
import os
import threading
import time
from ultralytics import YOLO
from datetime import datetime


# =========================================
# CONFIGURATION
# =========================================

CONFIDENCE_THRESHOLD = 0.65

# Warning distance in pixels
WARNING_BUFFER = 50

# Number of consecutive frames required
# before declaring a confirmed intrusion
RESTRICTED_CONFIRMATION_FRAMES = 5

# Number of frames a person can be missing before
# their loitering state is cleared.
TRACK_TIMEOUT_FRAMES = 30

# Number of seconds a person must remain in the warning zone
# before being classified as loitering
LOITERING_SECONDS = 10

# Number of processed frames used to calculate movement direction.
# A larger value makes direction more stable and avoids false STATIONARY
# results caused by tiny frame-to-frame movements.
DIRECTION_HISTORY_FRAMES = 8

# Minimum total movement (in pixels) across the history before deciding
# direction. This is deliberately small so it works for both video and
# live camera input.
MIN_DIRECTION_MOVEMENT = 5


# =========================================
# YOLO MODEL
# =========================================

model = YOLO("yolo11n.pt")


# =========================================
# VIDEO / CAMERA SOURCE
# =========================================

# Choose what you want to use:
# "LIVE"  -> phone/IP camera
# "VIDEO" -> prerecorded video file
SOURCE_TYPE = "LIVE"

# Used when SOURCE_TYPE = "LIVE"
LIVE_SOURCE = "http://10.115.51.4:8080/video"

# Used when SOURCE_TYPE = "VIDEO"
VIDEO_SOURCE = "border_test.mp4"


camera = cv2.VideoCapture(
    LIVE_SOURCE if SOURCE_TYPE == "LIVE" else VIDEO_SOURCE
)

if not camera.isOpened():
    print(
        f"Could not open {SOURCE_TYPE} source: "
        f"{LIVE_SOURCE if SOURCE_TYPE == 'LIVE' else VIDEO_SOURCE}"
    )
    exit()


# LIVE mode uses a background reader to avoid latency.
# VIDEO mode processes every frame in sequence.
latest_frame = None
frame_lock = threading.Lock()
camera_running = True
reader_thread = None


def camera_reader():
    global latest_frame, camera_running

    while camera_running:
        success, new_frame = camera.read()

        if not success:
            if SOURCE_TYPE == "LIVE":
                time.sleep(0.01)
                continue
            else:
                camera_running = False
                break

        with frame_lock:
            latest_frame = new_frame


if SOURCE_TYPE == "LIVE":

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    reader_thread = threading.Thread(
        target=camera_reader,
        daemon=True
    )
    reader_thread.start()

    while latest_frame is None and camera_running:
        time.sleep(0.01)

    if latest_frame is None:
        print("Could not receive frames from live camera.")
        camera_running = False
        camera.release()
        exit()

    with frame_lock:
        first_frame = latest_frame.copy()

    print("Live camera mode selected.")
    print("Low-latency camera reader started.")

else:

    success, first_frame = camera.read()

    if not success:
        print("Could not read the video file.")
        camera.release()
        exit()

    print(f"Video mode selected: {VIDEO_SOURCE}")


height, width = first_frame.shape[:2]

print(f"Source resolution: {width} x {height}")
print("")
# =========================================
# LOAD RESTRICTED ZONE
# =========================================

def load_zone():

    if os.path.exists("zones.json"):

        with open("zones.json", "r") as file:

            data = json.load(file)

            return data.get("restricted", [])

    return []


restricted_zone = load_zone()


# =========================================
# TRACKING STATE
# =========================================

person_states = {}
restricted_frames = {}
person_positions = {}

# Position history for each tracked person.
# This is used only for direction calculation and does not affect
# YOLO/ByteTrack tracking or live-camera frame handling.
person_position_history = {}

loiter_start_times = {}
loitering_status = {}
last_seen_frames = {}
restricted_alerted = {}

event_counter = 0


# =========================================
# WARNING MASK
# =========================================

warning_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)


# =========================================
# CREATE WARNING BUFFER
# =========================================

def create_warning_buffer():

    global warning_mask

    warning_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    if len(restricted_zone) < 3:
        return


    restricted_array = np.array(
        restricted_zone,
        dtype=np.int32
    )


    # Create restricted area mask

    restricted_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.fillPoly(
        restricted_mask,
        [restricted_array],
        255
    )


    # Create circular kernel

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            WARNING_BUFFER * 2 + 1,
            WARNING_BUFFER * 2 + 1
        )
    )


    # Expand restricted area

    expanded_mask = cv2.dilate(
        restricted_mask,
        kernel
    )


    # IMPORTANT:
    #
    # Remove the restricted area from the
    # warning area.
    #
    # Therefore:
    #
    # WARNING = expanded area - restricted area

    warning_mask = cv2.subtract(
        expanded_mask,
        restricted_mask
    )


# Create buffer for already-loaded zone

create_warning_buffer()


# =========================================
# POLYGON DRAWING
# =========================================

drawing = False


def mouse_callback(event, x, y, flags, param):

    global restricted_zone

    if event == cv2.EVENT_LBUTTONDOWN:

        if drawing:

            restricted_zone.append(
                [x, y]
            )

            print(
                f"Restricted point added: ({x}, {y})"
            )


# =========================================
# CREATE WINDOW
# =========================================

window_name = "Virtual Fence"

cv2.namedWindow(
    window_name
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


print("\nControls:")
print("R → Start drawing Restricted Zone")
print("F → Finish Polygon")
print("C → Clear Polygon")
print("S → Save Zone")
print("Q → Quit\n")

def get_border_direction(person_position, position_history, restricted_zone):
    """
    Calculate direction using movement over several processed frames
    instead of only comparing the current frame with the immediately
    previous frame.

    This makes direction detection more reliable for both prerecorded
    video and live camera streams.
    """

    if len(restricted_zone) < 3:
        return "UNKNOWN"

    if len(position_history) < 2:
        return "UNKNOWN"

    # Use the oldest and newest positions in the short history.
    previous_position = position_history[0]

    px, py = person_position
    prev_x, prev_y = previous_position

    movement_x = px - prev_x
    movement_y = py - prev_y

    movement_distance = np.sqrt(
        movement_x ** 2 + movement_y ** 2
    )

    # Only call a person stationary when they have genuinely moved
    # very little over several processed frames.
    if movement_distance < MIN_DIRECTION_MOVEMENT:
        return "STATIONARY"

    # Find the nearest point on the restricted boundary.
    contour = np.array(
        restricted_zone,
        dtype=np.float32
    )

    min_distance = float("inf")
    nearest_point = None

    for i in range(len(contour)):

        p1 = contour[i]
        p2 = contour[(i + 1) % len(contour)]

        line = p2 - p1

        line_length_squared = np.dot(
            line,
            line
        )

        if line_length_squared == 0:
            closest = p1

        else:
            t = np.dot(
                np.array([px, py]) - p1,
                line
            ) / line_length_squared

            t = max(0, min(1, t))

            closest = p1 + t * line

        distance = np.linalg.norm(
            np.array([px, py]) - closest
        )

        if distance < min_distance:
            min_distance = distance
            nearest_point = closest

    if nearest_point is None:
        return "UNKNOWN"

    # Direction from person toward the restricted boundary.
    to_boundary_x = nearest_point[0] - px
    to_boundary_y = nearest_point[1] - py

    boundary_distance = np.sqrt(
        to_boundary_x ** 2 +
        to_boundary_y ** 2
    )

    if boundary_distance == 0:
        return "AT_BOUNDARY"

    # Normalize movement vector.
    movement_x /= movement_distance
    movement_y /= movement_distance

    # Normalize boundary vector.
    to_boundary_x /= boundary_distance
    to_boundary_y /= boundary_distance

    # Dot product:
    # > 0.5  = moving toward boundary
    # < -0.5 = moving away
    # otherwise = moving sideways
    dot_product = (
        movement_x * to_boundary_x +
        movement_y * to_boundary_y
    )

    if dot_product > 0.5:
        return "APPROACHING"

    elif dot_product < -0.5:
        return "AWAY"

    else:
        return "SIDEWAYS"


# =========================================
# MAIN LOOP
# =========================================

while True:

    # LIVE mode: always process the newest available frame.
    # VIDEO mode: process every frame in sequence.
    if SOURCE_TYPE == "LIVE":

        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

    else:

        # Use the first frame already read before the loop.
        if "video_first_frame_used" not in locals():
            frame = first_frame.copy()
            video_first_frame_used = True
        else:
            success, frame = camera.read()

            if not success:
                print("Video finished.")
                break



    # =====================================
    # PERSON DETECTION + TRACKING
    # =====================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # =====================================
    # PROCESS PERSONS
    # =====================================

    for result in results:

        if result.boxes.id is None:
            continue


        boxes = result.boxes


        track_ids = (
            boxes.id
            .int()
            .cpu()
            .tolist()
        )


        coordinates = (
            boxes.xyxy
            .int()
            .cpu()
            .tolist()
        )


        confidences = (
            boxes.conf
            .cpu()
            .tolist()
        )


        # =================================
        # PROCESS EACH PERSON
        # =================================

        for track_id, box, confidence in zip(
            track_ids,
            coordinates,
            confidences
        ):

            # Mark this tracked person as currently visible.
            last_seen_frames[track_id] = 0

            x1, y1, x2, y2 = box


            # =================================
            # GROUND POSITION
            # =================================

            person_x = int(
                (x1 + x2) / 2
            )


            person_y = int(
                y2 - (y2 - y1) * 0.05
            )

            # =================================
            # KEEP POINT SAFELY INSIDE IMAGE
            # =================================

            person_x = max(
                0,
                min(width - 1, person_x)
            )

            person_y = max(
                0,
                min(height - 1, person_y)
            )


            # =================================
            # TRACK PERSON POSITION
            # =================================

            current_position = (
                person_x,
                person_y
            )

            # =================================
            # UPDATE POSITION HISTORY
            # =================================

            if track_id not in person_position_history:
                person_position_history[track_id] = []

            person_position_history[track_id].append(
                current_position
            )

            # Keep only a short history.
            if len(person_position_history[track_id]) > DIRECTION_HISTORY_FRAMES:
                person_position_history[track_id].pop(0)


            # =================================
            # CALCULATE BORDER DIRECTION
            # =================================

            border_direction = get_border_direction(
                current_position,
                person_position_history[track_id],
                restricted_zone
            )


            # =================================
            # STORE CURRENT POSITION
            # =================================

            person_positions[track_id] = current_position

            # =================================
            # DEFAULT ZONE
            # =================================

            current_person_zone = "NORMAL"


            # =================================
            # RESTRICTED ZONE CHECK
            # =================================

            if len(restricted_zone) >= 3:

                restricted_array = np.array(
                    restricted_zone,
                    dtype=np.int32
                )


                inside_restricted = cv2.pointPolygonTest(
                    restricted_array,
                    (person_x, person_y),
                    False
                )


                if inside_restricted >= 0:

                    current_person_zone = "RESTRICTED"


            # =================================
            # WARNING ZONE CHECK
            # =================================

            if current_person_zone != "RESTRICTED":

                if warning_mask[person_y, person_x] > 0:

                    current_person_zone = "WARNING"


            # =================================
            # LOITERING DETECTION
            # =================================

            # =================================
            # LOITERING DETECTION
            # =================================

            # Keep loitering state for this track once it is detected.
            # It is cleared only when the track is lost for TRACK_TIMEOUT_FRAMES.
            loitering = loitering_status.get(track_id, False)

            if current_person_zone == "WARNING":

                # Start the timer only if this track has never started
                # a warning-zone timer.
                if track_id not in loiter_start_times:
                    loiter_start_times[track_id] = datetime.now()

                elapsed_time = (
                    datetime.now()
                    - loiter_start_times[track_id]
                ).total_seconds()

                if elapsed_time >= LOITERING_SECONDS:
                    loitering_status[track_id] = True
                    loitering = True


            # =================================
            # RESTRICTED ZONE CONFIRMATION
            # =================================

            if current_person_zone == "RESTRICTED":

                restricted_frames[track_id] = (
                    restricted_frames.get(track_id, 0) + 1
                )

            else:

                restricted_frames[track_id] = 0

                # Leaving the restricted zone re-arms the intrusion alert.
                # A later confirmed re-entry can therefore create a new event.
                restricted_alerted[track_id] = False


            # =================================
            # CREATE INTRUSION EVENT
            # =================================

            if (
                restricted_frames[track_id]
                >= RESTRICTED_CONFIRMATION_FRAMES
                and not restricted_alerted.get(track_id, False)
            ):

                event_counter += 1


                event = {

                    "event_id":
                        f"EVT-{event_counter:06d}",

                    "event_type":
                        "RESTRICTED_INTRUSION",

                    "track_id":
                        int(track_id),

                    "zone":
                        "RESTRICTED",

                    "timestamp":
                        datetime.now().isoformat(),

                    "confidence":
                        float(confidence)
                }


                print("\n🚨 INTRUSION EVENT")

                print(event)

                # Prevent repeated events while the same person
                # continuously remains inside the restricted zone.
                restricted_alerted[track_id] = True


            # =================================
            # STORE CURRENT STATE
            # =================================

            person_states[track_id] = (
                current_person_zone
            )


            # =================================
            # BOX COLOR
            # =================================

            if current_person_zone == "RESTRICTED":

                box_color = (
                    0,
                    0,
                    255
                )

            elif current_person_zone == "WARNING":

                box_color = (
                    0,
                    255,
                    255
                )

            else:

                box_color = (
                    0,
                    255,
                    0
                )


            # =================================
            # DRAW PERSON
            # =================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # =================================
            # DRAW LABEL
            # =================================

            # Direction is useful while approaching the restricted zone,
            # but once the person is inside the restricted zone, show
            # the intrusion status instead of a direction.
            if current_person_zone == "RESTRICTED":

                label = (
                    f"Person #{track_id} | "
                    f"RESTRICTED"
                )

            elif loitering:

                label = (
                    f"Person #{track_id} | "
                    f"{current_person_zone} | "
                    f"{border_direction} | "
                    f"LOITERING"
                )

            else:

                label = (
                    f"Person #{track_id} | "
                    f"{current_person_zone} | "
                    f"{border_direction}"
                )


            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )


            # =================================
            # DRAW GROUND POINT
            # =================================

            cv2.circle(
                frame,
                (person_x, person_y),
                6,
                box_color,
                -1
            )


    # =====================================
    # CLEAN UP LOST TRACKS
    # =====================================

    current_track_ids = set()

    for result in results:

        if result.boxes.id is not None:

            current_track_ids.update(
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )


    for track_id in list(last_seen_frames.keys()):

        if track_id in current_track_ids:

            last_seen_frames[track_id] = 0

        else:

            last_seen_frames[track_id] += 1


            if last_seen_frames[track_id] > TRACK_TIMEOUT_FRAMES:

                # The person has been absent long enough
                # that their old loitering state should
                # no longer be reused.

                loiter_start_times.pop(
                    track_id,
                    None
                )

                loitering_status.pop(
                    track_id,
                    None
                )

                person_positions.pop(
                    track_id,
                    None
                )

                person_position_history.pop(
                    track_id,
                    None
                )

                person_states.pop(
                    track_id,
                    None
                )

                restricted_frames.pop(
                    track_id,
                    None
                )

                restricted_alerted.pop(
                    track_id,
                    None
                )

                last_seen_frames.pop(
                    track_id,
                    None
                )


    # =====================================
    # DRAW WARNING AREA
    # =====================================

    if len(restricted_zone) >= 3:

        warning_display = np.zeros_like(
            frame
        )


        warning_display[
            warning_mask > 0
        ] = (
            0,
            255,
            255
        )


        frame = cv2.addWeighted(
            frame,
            1.0,
            warning_display,
            0.12,
            0
        )


        # Find warning boundary

        contours, _ = cv2.findContours(
            warning_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        for contour in contours:

            # Do not draw contours that touch
            # the camera boundary.
            #
            # This prevents ugly yellow lines
            # appearing along the frame edge.

            touches_edge = False


            for point in contour:

                x, y = point[0]


                if (
                    x <= 0
                    or x >= width - 1
                    or y <= 0
                    or y >= height - 1
                ):

                    touches_edge = True

                    break


            if not touches_edge:

                cv2.drawContours(
                    frame,
                    [contour],
                    -1,
                    (0, 255, 255),
                    2
                )


    # =====================================
    # DRAW RESTRICTED ZONE
    # =====================================

    if len(restricted_zone) > 0:

        points = np.array(
            restricted_zone,
            dtype=np.int32
        )


        # Draw points while editing

        for point in restricted_zone:

            cv2.circle(
                frame,
                tuple(point),
                6,
                (0, 0, 255),
                -1
            )


        # Draw completed polygon

        if len(points) >= 3:

            overlay = frame.copy()


            cv2.fillPoly(
                overlay,
                [points],
                (0, 0, 255)
            )


            frame = cv2.addWeighted(
                overlay,
                0.20,
                frame,
                0.80,
                0
            )


            cv2.polylines(
                frame,
                [points],
                True,
                (0, 0, 255),
                3
            )


    # =====================================
    # DISPLAY INSTRUCTIONS
    # =====================================

    cv2.putText(
        frame,
        "R: Restricted | F: Finish | C: Clear | S: Save | Q: Quit",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # =====================================
    # DISPLAY FRAME
    # =====================================

    cv2.imshow(
        window_name,
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # =====================================
    # START DRAWING
    # =====================================

    if key == ord("r"):

        restricted_zone = []

        drawing = True

        warning_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        print(
            "Drawing RESTRICTED zone..."
        )


    # =====================================
    # FINISH POLYGON
    # =====================================

    elif key == ord("f"):

        if len(restricted_zone) >= 3:

            drawing = False

            create_warning_buffer()

            print(
                "Restricted zone completed."
            )

            print(
                f"Warning buffer: "
                f"{WARNING_BUFFER} pixels"
            )

        else:

            print(
                "Need at least 3 points."
            )


    # =====================================
    # CLEAR
    # =====================================

    elif key == ord("c"):

        restricted_zone = []

        warning_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        drawing = False

        print(
            "Restricted zone cleared."
        )


    # =====================================
    # SAVE
    # =====================================

    elif key == ord("s"):

        data = {
            "restricted": restricted_zone
        }


        with open(
            "zones.json",
            "w"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )


        print(
            "Restricted zone saved."
        )


    # =====================================
    # QUIT
    # =====================================

    elif key == ord("q"):

        break


# =========================================
# CLEANUP
# =========================================

camera_running = False

if reader_thread is not None:
    reader_thread.join(timeout=1.0)

camera.release()

cv2.destroyAllWindows()