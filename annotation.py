import cv2


def annotate(frame, detections, text):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{d['label']} {d['confidence']:.2f}"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        label_x = max(0, min(x1, frame.shape[1] - text_width - 12))
        label_y = max(text_height + 10, min(y1 - 6, frame.shape[0] - baseline - 6))
        cv2.rectangle(
            frame,
            (label_x, label_y - text_height - 6),
            (label_x + text_width + 12, label_y + baseline + 5),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x + 6, label_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
