import cv2


def write_cells(img, cell_position):
    for rec in cell_position:
        cv2.rectangle(img, rec[0], rec[-1], (255, 255, 255), 2)

    cv2.imwrite("cdst2.jpg", img)
