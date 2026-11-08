"""Pure committed-group accounting for the proposed finite accumulation study.

This validates serialized evidence; it does not publish snapshots, perform
optimizer steps, change public labels or admit training.
"""
import math
import re
import json

from accumulation_contract import IMAGE_BUDGET, POLICIES, apply_group, committed_boundary, group_specs, update_count

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def require_optimizer_ownership(model, optimizer):
    """Require exact identity ownership, independent of parameter group order."""
    parameters = list(model.parameters())
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if (not parameters or len(owned) != len(parameters) or len({id(p) for p in owned}) != len(owned) or
            {id(p) for p in owned} != {id(p) for p in parameters}):
        raise ValueError("optimizer must own every model parameter exactly once")


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def group_entry(policy, update, recipes, *, mean_loss, gradient_norm):
    """Bind each actual serialized public-observer recipe to its expected cursor.

    The independent curator must additionally re-render/reobserve each recipe;
    this function does not infer a type or status from the fixture index.
    """
    expected = group_specs(policy, update)
    if type(recipes) not in (tuple, list) or len(recipes) != len(expected):
        raise ValueError("exactly one complete group of recipes required")
    if not _finite_number(mean_loss) or not _finite_number(gradient_norm) or gradient_norm < 0:
        raise ValueError("finite loss and nonnegative gradient norm required")
    image_rows = []
    curriculum = "all-class-interleaved-v1" if POLICIES[policy][1] else "all-class-fit-v1"
    for (image, index, view), recipe in zip(expected, recipes):
        if (type(recipe) is not dict or recipe.get("index") != index or type(recipe.get("index")) is not int or
                recipe.get("view") != view or type(recipe.get("view")) is not int or
                recipe.get("sample_id") != f"all-class-{index:02d}-v{view}" or
                recipe.get("group_id") != f"all-class-{index:02d}" or recipe.get("split") != "fit" or
                recipe.get("curriculum") != curriculum or recipe.get("profile_square") is not False or
                not isinstance(recipe.get("image_pixels_sha256"), str) or not _SHA.fullmatch(recipe["image_pixels_sha256"]) or
                not isinstance(recipe.get("state_sha256"), str) or not _SHA.fullmatch(recipe["state_sha256"])):
            raise ValueError("actual public recipe differs from the expected image cursor")
        # Freeze a JSON-stable copy; later mutation of a renderer record cannot
        # silently alter the evidence awaiting a boundary snapshot.
        recipe = json.loads(json.dumps(recipe, allow_nan=False))
        image_rows.append({"image": image, "recipe": recipe})
    return {"policy": policy, "update": update, "images": image_rows,
            "mean_loss": mean_loss, "gradient_norm": gradient_norm}


def execute_group(model, optimizer, policy, update, recipes, loss_for):
    """Validate inputs, then perform one whole update and return its journal entry.

    A driver may append/commit only the returned entry. If the optimizer or
    post-step reporting raises, the process must abort without publishing a
    snapshot; this function does not promise rollback of a failed optimizer.
    """
    preflight = group_entry(policy, update, recipes, mean_loss=0., gradient_norm=0.)
    require_optimizer_ownership(model, optimizer)
    frozen_recipes = [row["recipe"] for row in preflight["images"]]
    outcome = apply_group(model, optimizer, frozen_recipes, loss_for)
    if (type(outcome) is not dict or set(outcome) != {"mean_loss", "gradient_norm", "images"} or
            type(outcome["images"]) is not int or outcome["images"] != len(frozen_recipes)):
        raise ValueError("complete optimizer outcome required")
    return group_entry(policy, update, frozen_recipes,
                       mean_loss=outcome["mean_loss"], gradient_norm=outcome["gradient_norm"])


def validate_ledger(policy, entries):
    """Validate a complete-update prefix and derive explicit image/update cursors."""
    maximum = update_count(policy)
    if type(entries) is not list or len(entries) > maximum:
        raise ValueError("finite list of committed groups required")
    for expected_update, entry in enumerate(entries, start=1):
        if type(entry) is not dict or set(entry) != {"policy", "update", "images", "mean_loss", "gradient_norm"}:
            raise ValueError("exact committed group schema required")
        if entry["policy"] != policy or type(entry["update"]) is not int or entry["update"] != expected_update:
            raise ValueError("group policy or update cursor differs")
        images = entry["images"]
        if type(images) is not list or any(type(row) is not dict or set(row) != {"image", "recipe"} for row in images):
            raise ValueError("exact image record schema required")
        canonical = group_entry(policy, expected_update, [row["recipe"] for row in images],
                                mean_loss=entry["mean_loss"], gradient_norm=entry["gradient_norm"])
        if entry != canonical or any(type(row["image"]) is not int for row in images):
            raise ValueError("group contents or image cursor differ")
    images_seen = len(entries)*POLICIES[policy][0]
    if committed_boundary(policy, images_seen) != len(entries):
        raise ValueError("non-boundary ledger")
    complete = images_seen == IMAGE_BUDGET
    return {"policy": policy, "images_seen": images_seen, "optimizer_updates": len(entries),
            "target_images": IMAGE_BUDGET, "target_updates": maximum,
            "status": "complete" if complete else "partial",
            "next_image": None if complete else images_seen+1,
            "next_update": None if complete else len(entries)+1}


def verify_resume_cursor(policy, entries, *, snapshot_images, snapshot_updates):
    """Reject mismatched or partial-group state before any resumed optimization."""
    state = validate_ledger(policy, entries)
    if (type(snapshot_images) is not int or type(snapshot_updates) is not int or
            snapshot_images != state["images_seen"] or snapshot_updates != state["optimizer_updates"] or
            committed_boundary(policy, snapshot_images) != snapshot_updates):
        raise ValueError("snapshot and complete-group ledger cursors differ")
    return state
