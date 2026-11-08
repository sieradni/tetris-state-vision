"""Strict output comparison for the conditional type-head intervention."""


def verify_non_type_outputs(original, changed, torch):
    """Compare every output byte except the explicitly identified type logits.

    The caller must separately verify head inputs, model state and input identity.
    Both stage and final references are checked, even when they alias.
    """
    if set(original) != {"stages", "final"} or set(changed) != set(original):
        raise ValueError("unexpected model output structure")
    if not original["stages"] or len(original["stages"]) != len(changed["stages"]):
        raise ValueError("stage count differs")

    def compare(a, b, path, stage=False):
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise ValueError(f"mapping differs: {path}")
            for key in a:
                if key != "layer_type" or not stage:
                    compare(a[key], b[key], f"{path}.{key}")
                elif not (torch.is_tensor(a[key]) and torch.is_tensor(b[key]) and
                          a[key].shape == b[key].shape and a[key].dtype == b[key].dtype):
                    raise ValueError(f"type output structure differs: {path}")
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError(f"sequence differs: {path}")
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f"{path}[{i}]")
        elif torch.is_tensor(a):
            if (not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape or
                    a.device != b.device or not torch.equal(a.contiguous().view(torch.uint8),
                                                            b.contiguous().view(torch.uint8))):
                raise ValueError(f"tensor bytes differ: {path}")
        else:
            raise ValueError(f"unexpected output value: {path}")

    for index, (a, b) in enumerate(zip(original["stages"], changed["stages"])):
        if "layer_type" not in a or "layer_type" not in b:
            raise ValueError("missing type logits")
        compare(a, b, f"stages[{index}]", stage=True)
    if "layer_type" not in original["final"] or "layer_type" not in changed["final"]:
        raise ValueError("missing final type logits")
    compare(original["final"], changed["final"], "final", stage=True)
