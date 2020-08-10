

def line_equation_coefficients(p1, p2):
    """
    Ax + By + C = 0
    
    >>> line_equation_coefficients((0,10), (0,0))
    # 10, 0, 0
    
    Args:
        p1: first point like (x,y)
        p2: second point like (x,y)

    Returns:
        line equation coefficients
    """

    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    return A, B, -C


def line_intersection(l1, l2):
    """
    Find two line intersection point or return false if such point does not exist
    """
    D = l1[0] * l2[1] - l1[1] * l2[0]
    Dx = l1[2] * l2[1] - l1[1] * l2[2]
    Dy = l1[0] * l2[2] - l1[2] * l2[0]

    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    else:
        return False