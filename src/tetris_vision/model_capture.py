"""Synthetic capture at model resolution with labels observed after that resize.

This creates a new final-capture recipe, not a claim of bit equivalence to an
earlier PNG followed by another resize. Latent render inputs stay outside the
model and outside the returned image-supported state. No real-image relabeling.
"""
from dataclasses import replace

from .layout import Layout
from .render import Appearance, Scene, render
from .state import check


def model_capture_style(style, layout, *, max_side=512):
    """Plan one aspect-preserving final resize using actual rounded dimensions."""
    check(isinstance(style, Appearance) and isinstance(layout, Layout), "validated capture appearance/layout required")
    check(type(max_side) is int and 64 <= max_side <= 1536, "model capture side must be64..1536")
    left, top, right, bottom = style.crop or (0, 0, *layout.canvas)
    check(0 <= left < right <= layout.canvas[0] and 0 <= top < bottom <= layout.canvas[1], "capture crop outside source")
    width, height = style.output_size or (right-left, bottom-top)
    scale = min(1., max_side/max(width, height))
    size = (max(1, round(width*scale)), max(1, round(height*scale)))
    # Appearance's existing renderer capacity excludes sub64 output axes. Do
    # not distort aspect or silently enlarge the canvas to evade the budget.
    check(min(size) >= 64, "capture aspect/side produces an axis below renderer capacity64")
    return replace(style, output_size=size)


def render_for_model(scene, style=Appearance(), layout=Layout(), *, max_side=512):
    """Render/reobserve once, so prepare_image(max_side) need not resize again.

    For a reproducible ledger, record model_capture_style(style, layout) as the
    effective appearance. The existing observer owns all visibility/anchor rules;
    source component identities are never copied across the resize boundary.
    """
    check(isinstance(scene, Scene), "validated synthetic scene required")
    effective = model_capture_style(style, layout, max_side=max_side)
    result = render(scene, effective, layout)
    check(result.image.size == effective.output_size and max(result.image.size) <= max_side,
          "final capture/model budget differs")
    return result
