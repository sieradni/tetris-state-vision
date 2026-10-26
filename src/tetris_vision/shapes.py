"""Canonical mino shapes, with y increasing upward like board rows."""
SHAPES = {
    "I": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "O": ((0, 0), (1, 0), (0, 1), (1, 1)),
    "T": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "S": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "Z": ((1, 0), (2, 0), (0, 1), (1, 1)),
    "J": ((0, 0), (1, 0), (2, 0), (0, 1)),
    "L": ((0, 0), (1, 0), (2, 0), (2, 1)),
}


def normalize(cells):
    cells = tuple(cells)
    left = min(c for c, _ in cells)
    bottom = min(r for _, r in cells)
    return tuple(sorted((c - left, r - bottom) for c, r in cells))


def is_piece_shape(piece, cells):
    candidate = normalize(cells)
    shape = SHAPES[piece]
    for _ in range(4):
        if candidate == normalize(shape):
            return True
        shape = tuple((-r, c) for c, r in shape)
    return False
