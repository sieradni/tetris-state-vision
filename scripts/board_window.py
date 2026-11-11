"""One derived window for every consumer of a per-cell occupancy-style map.

`final["occupancy"]` (and `observability`, `semantic`) is
`(batch, visible_rows+spawn_rows, columns[, classes])`: one channel per board cell plus the unlabelled
spawn margin. The labelled board is the first `visible_rows` rows of the map, and every consumer sizes that
window from the declared config (or from the target's own shape) and refuses a map that cannot cover it.
The window is never a literal, so a structural change to the board height or the spawn margin cannot
silently truncate a read (V-089).
"""


def _field(config, name):
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def board_window(config):
    """The declared `(visible rows, spawn rows, columns)` of one board map."""
    values = tuple(_field(config, name) for name in ("visible_rows", "spawn_rows", "columns"))
    for value, name in zip(values, ("visible_rows", "spawn_rows", "columns")):
        if type(value) is not int or value < 0 or (name != "spawn_rows" and value == 0):
            raise ValueError("declared %s required, got %r" % (name, value))
    return values


def board_shape(tensor, name="map"):
    """`(batch, rows, columns)` of a `(batch, rows, columns)` or `(batch, rows, columns, classes)` map."""
    shape = tuple(int(value) for value in tensor.shape)
    if tensor.dim() == 3:
        return shape[0], shape[1], shape[2]
    if tensor.dim() == 4:
        return shape[0], shape[1], shape[2]
    raise ValueError("%s must be (batch, rows, columns[, classes]), got %s" % (name, shape))


def visible_board(tensor, config, name="map"):
    """The visible-board rows of a map that must match the declared board exactly."""
    rows, spawn, columns = board_window(config)
    batch, produced_rows, produced_columns = board_shape(tensor, name)
    if batch != 1 or produced_rows != rows+spawn or produced_columns != columns:
        raise ValueError("%s does not match the declared %dx%d board plus %d spawn rows: %s"
                         % (name, rows, columns, spawn, tuple(tensor.shape)))
    return tensor[:, :rows]


def visible_board_for_target(tensor, target, name="map"):
    """The labelled visible board of a map, sized by the target rather than the config."""
    rows, columns = (int(value) for value in target["occupied"].shape[-2:])
    _, produced_rows, produced_columns = board_shape(tensor, name)
    if produced_columns != columns or produced_rows < rows:
        raise ValueError("%s cannot cover the labelled %dx%d board: %s"
                         % (name, rows, columns, tuple(tensor.shape)))
    return tensor[:, :rows]
