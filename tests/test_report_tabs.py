from coinfosim.reports.html_tabs import TAB_CSS, TAB_JS, tab_group


def test_default_tab_is_active_and_other_panels_are_hidden():
    rendered = tab_group(
        "metric-tabs",
        [("first", "First", "<p>first panel</p>"),
         ("second", "Second", "<p>second panel</p>")],
        "second",
    )

    assert "class='tab-btn active' data-group='metric-tabs' data-key='second'" in rendered
    assert "data-key='second' style='display:block'" in rendered
    assert "data-key='first' style='display:none'" in rendered
    assert rendered.count("style='display:block'") == 1


def test_nested_groups_have_independent_deterministic_ids():
    inner = tab_group(
        "outer-real-projection",
        [("1d", "1D", "one"), ("2d", "2D", "two")],
        "1d",
    )
    rendered = tab_group(
        "outer-source",
        [("real", "Real", inner), ("gmm", "GMM", "other")],
        "real",
    )

    assert "id='tabs-outer-source'" in rendered
    assert "id='tabs-outer-real-projection'" in rendered
    assert "data-group='outer-source'" in rendered
    assert "data-group='outer-real-projection'" in rendered


def test_labels_keys_and_group_ids_are_escaped_but_panel_html_is_not():
    rendered = tab_group(
        "group'&",
        [("key'<&", "Label <&", "<strong>rendered</strong>")],
        "key'<&",
    )

    assert "group&#x27;&amp;" in rendered
    assert "key&#x27;&lt;&amp;" in rendered
    assert "Label &lt;&amp;" in rendered
    assert "<strong>rendered</strong>" in rendered


def test_one_shared_payload_controls_nested_groups_without_selects():
    inner = tab_group("inner", [("a", "A", "A"), ("b", "B", "B")], "a")
    document = (
        f"<style>{TAB_CSS}</style>"
        + tab_group("outer", [("x", "X", inner), ("y", "Y", "Y")], "x")
        + TAB_JS
    )

    assert document.count("<script>") == 1
    assert "querySelectorAll('.tab-btn')" in document
    assert ".tab-bar" in document
    assert ".tab-btn.active" in document
    assert "<select" not in document.lower()
