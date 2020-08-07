import cv2


def write_cells(img, cell_position):
    for rec in cell_position:
        cv2.rectangle(img, rec[0], rec[-1], (255, 255, 255), 2)

    cv2.imwrite("cells.jpg", img)


def write_dots(img, positions):
    for pos in positions:
        cv2.circle(img, tuple(pos), 10, (255, 0, 0), 2)

    cv2.imwrite("cells.jpg", img)


def write_lines(img, lines):
    for l in lines:
        # if type(l) is list:
        l = l[0]

        cv2.line(img, (l[0], l[1]), (l[2], l[3]), (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imwrite("lines.jpg", img)