# copyright © All rights reserved to Eli-M-Fisher
# Can be used freely (:
import cv2
import numpy as np

# ASCII density string
density = "Ñ@#W$9876543210?!abc;:+=-,._          "


def main():
    cap = cv2.VideoCapture(0)

    # "Text resolution" of the ASCII version
    ascii_width, ascii_height = 80, 60

    # Character drawing params
    font_scale = 0.35
    font_thickness = 1
    char_width_px = 10
    char_height_px = 12

    # Scale factors for final display
    ascii_scale = 2.0  # Enlarge the ASCII image
    original_scale = 0.3  # Shrink the camera feed

    cv2.namedWindow("Live + ASCII", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ----------------------------------------------------------------------
        # 1) Build the base ASCII image from the current frame
        # ----------------------------------------------------------------------
        # Resize the frame to ascii_width x ascii_height
        small_frame = cv2.resize(frame, (ascii_width, ascii_height))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Create a black background to draw ASCII text
        ascii_image = np.zeros(
            (ascii_height * char_height_px, ascii_width * char_width_px, 3),
            dtype=np.uint8
        )

        # Convert each pixel to an ASCII character
        for j in range(ascii_height):
            for i in range(ascii_width):
                pixel_value = gray[j, i]
                # Map the grayscale to an index in the density string
                idx = int(np.interp(pixel_value, [0, 255], [0, len(density) - 1]))
                char = density[idx]

                x_pos = i * char_width_px
                y_pos = j * char_height_px

                # Draw the character
                cv2.putText(
                    ascii_image,
                    char,
                    (x_pos, y_pos + char_height_px - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # white
                    font_thickness,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------------------------
        # 2) Enlarge the ASCII image
        # ----------------------------------------------------------------------
        new_w = int(ascii_image.shape[1] * ascii_scale)
        new_h = int(ascii_image.shape[0] * ascii_scale)
        ascii_image_big = cv2.resize(ascii_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # ----------------------------------------------------------------------
        # 3) Shrink the original camera feed
        # ----------------------------------------------------------------------
        small_original = cv2.resize(
            frame,
            None,
            fx=original_scale,
            fy=original_scale,
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------------------------
        # 4) Match heights and stack side-by-side
        # ----------------------------------------------------------------------
        # The ASCII image height
        ascii_h = ascii_image_big.shape[0]
        # The original feed height
        orig_h = small_original.shape[0]

        # If needed, we add black padding so both have the same height
        if ascii_h > orig_h:
            # ASCII is taller; pad the original
            padding = np.zeros((ascii_h - orig_h, small_original.shape[1], 3), dtype=np.uint8)
            small_original = np.vstack((small_original, padding))
        elif orig_h > ascii_h:
            # Original is taller; pad the ASCII
            padding = np.zeros((orig_h - ascii_h, ascii_image_big.shape[1], 3), dtype=np.uint8)
            ascii_image_big = np.vstack((ascii_image_big, padding))

        # Combine side by side (original on the left, ASCII on the right)
        combined = np.hstack((small_original, ascii_image_big))

        # ----------------------------------------------------------------------
        # 5) Display the combined image
        # ----------------------------------------------------------------------
        cv2.imshow("Live + ASCII", combined)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
