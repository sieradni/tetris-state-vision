"""Pure analysis of supported role logits and final linear-head CE gradients."""
import math


def analyze(points):
    if len(points) != 17 or [p["step"] for p in points] != list(range(0, 1025, 64)):
        raise ValueError("fixed 17-checkpoint timeline required")
    timeline, transitions, previous, previous_head = [], [], None, None
    identities, targets = None, None
    for point in points:
        cases = point["cases"]
        if len(cases) != 32:
            raise ValueError("fixed 32-image population required")
        current_ids = [(c["sample_id"], c["group_id"]) for c in cases]
        if len(set(current_ids)) != 32 or identities is not None and current_ids != identities:
            raise ValueError("fixture identity/order changed")
        identities = current_ids
        correct, observed_targets, class_vectors, counts = {}, {}, {i: [] for i in range(8)}, [0]*8
        for case in cases:
            if len(case["features"]) != 2 or len(case["logits"]) != 2:
                raise ValueError("two role observations required")
            for index, role in enumerate(("active", "ghost")):
                item, x, logits = case["roles"][role], case["features"][index], case["logits"][index]
                if (item["target_known"] is not True or type(item["target_type"]) is not int or
                        not 0 <= item["target_type"] < 8 or len(x) != 64 or len(logits) != 8 or
                        any(not math.isfinite(v) for v in x + logits)):
                    raise ValueError("known finite supported role vectors required")
                target = item["target_type"]
                observed_targets[(case["sample_id"], role)] = (target, item["target_pose_known"])
                predicted = max(range(8), key=logits.__getitem__)
                hit = predicted == target
                if item["type_correct"] is not hit:
                    raise ValueError("saved type correctness differs from logits")
                correct[(case["sample_id"], role)] = hit
                maximum = max(logits)
                exp = [math.exp(v-maximum) for v in logits]
                probabilities = [v/math.fsum(exp) for v in exp]
                delta = [p - (i == target) for i, p in enumerate(probabilities)]
                # Affine head only; unweighted per-role CE gradient, no Adam,
                # clipping, auxiliary stages or feature derivatives.
                vector = [d*v for d in delta for v in (*x, 1.0)]
                class_vectors[target].append(vector)
                counts[target] += 1
        if counts != [8]*8:
            raise ValueError("balanced supported class population required")
        if targets is not None and observed_targets != targets:
            raise ValueError("public target changed across checkpoints")
        targets = observed_targets
        means = [[math.fsum(values)/8 for values in zip(*class_vectors[label])] for label in range(8)]
        norms = [math.sqrt(math.fsum(v*v for v in vec)) for vec in means]
        cosines = [[(math.fsum(x*y for x, y in zip(a, b))/(norms[i]*norms[j]))
                    if norms[i] and norms[j] else None for j, b in enumerate(means)] for i, a in enumerate(means)]
        if previous is not None:
            transitions.append({"from": point["step"]-64, "to": point["step"],
                                "correct_to_wrong": sum(previous[k] and not correct[k] for k in correct),
                                "wrong_to_correct": sum(not previous[k] and correct[k] for k in correct)})
        head = point["head"]
        flattened = [v for row in head["weight"] for v in row] + head["bias"]
        if len(flattened) != 8*64+8 or any(not math.isfinite(v) for v in flattened):
            raise ValueError("finite shared 8x64 affine head required")
        movement = None if previous_head is None else math.sqrt(math.fsum((a-b)**2 for a, b in zip(flattened, previous_head)))
        timeline.append({"step": point["step"], "correct_types": sum(correct.values()),
                         "class_gradient_norms": norms, "class_gradient_cosines": cosines,
                         "negative_class_pairs": sum(cosines[i][j] is not None and cosines[i][j] < 0
                                                     for i in range(8) for j in range(i+1, 8)),
                         "head_parameter_l2_movement": movement})
        previous, previous_head = correct, flattened
    return {"timeline": timeline, "transitions": transitions,
            "scope": "64 supported roles on 32 dependent synthetic views. Per-class mean unweighted final-head cross-entropy gradient with captured features fixed; cosine is not actual joint optimizer or feature-gradient interference. Checkpoint transitions are 64 updates apart; intervening changes are unobserved."}
