"""Shared static tab controls for offline HTML reports."""

from __future__ import annotations

import html
from typing import Sequence, Tuple

Tab = Tuple[str, str, str]


TAB_CSS = """
  /* Static tab selectors (buttons switch panels; no external JS libs). */
  .tab-bar { display: flex; flex-wrap: wrap; gap: .3rem; margin: .7rem 0 0 0;
             border-bottom: 2px solid #ddd; }
  .tab-btn { border: 1px solid #c9d4e0; background: #eef2f7; color: #234;
    padding: .35rem .95rem; border-radius: 6px 6px 0 0; cursor: pointer;
    font-size: .9rem; }
  .tab-btn:hover { background: #dde7f1; }
  .tab-btn.active { background: #1f77b4; color: #fff; border-color: #1f77b4;
                    font-weight: 600; }
  .tab-panel { padding: .5rem 0; }
  .selector-caption { color: #555; font-size: .9rem; margin: .35rem 0 .7rem; }
"""


TAB_JS = """
<script>(function(){
  var btns = document.querySelectorAll('.tab-btn');
  btns.forEach(function(btn){
    btn.addEventListener('click', function(){
      var g = btn.getAttribute('data-group');
      var k = btn.getAttribute('data-key');
      document.querySelectorAll('.tab-btn').forEach(function(b){
        if (b.getAttribute('data-group') === g) {
          b.classList.toggle('active', b.getAttribute('data-key') === k);
        }
      });
      document.querySelectorAll('.tab-panel').forEach(function(p){
        if (p.getAttribute('data-group') === g) {
          p.style.display = (p.getAttribute('data-key') === k) ? 'block' : 'none';
        }
      });
    });
  });
})();</script>
"""


def tab_group(group_id: str, tabs: Sequence[Tab], default_key: str) -> str:
    """Return a static tab group controlled by :data:`TAB_JS`.

    ``tabs`` contains ``(key, visible_label, rendered_panel_html)`` tuples.
    Labels and attribute values are escaped; panel HTML is intentionally kept
    intact. Callers must give every group in a document a unique ``group_id``.
    """

    group_id = str(group_id)
    normalized = [(str(key), str(label), panel) for key, label, panel in tabs]
    if not normalized:
        raise ValueError("tab group must contain at least one tab")
    keys = [key for key, _, _ in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("tab keys must be unique within a group")
    default_key = str(default_key)
    if default_key not in keys:
        raise ValueError("default_key must identify a tab in the group")

    escaped_group = html.escape(group_id, quote=True)
    buttons = "".join(
        f"<button type='button' class='tab-btn{' active' if key == default_key else ''}' "
        f"data-group='{escaped_group}' data-key='{html.escape(key, quote=True)}'>"
        f"{html.escape(label)}</button>"
        for key, label, _ in normalized
    )
    panels = "".join(
        f"<div class='tab-panel' data-group='{escaped_group}' "
        f"data-key='{html.escape(key, quote=True)}' "
        f"style='display:{'block' if key == default_key else 'none'}'>{panel}</div>"
        for key, _, panel in normalized
    )
    return (
        f"<div class='tabs' id='tabs-{escaped_group}'>"
        f"<div class='tab-bar'>{buttons}</div>{panels}</div>"
    )
