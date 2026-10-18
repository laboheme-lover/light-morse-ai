from light_table import CLASS_CODES, class_for_label, code_for_label, units_for_char


def test_65_classes_with_unique_codes():
    assert len(CLASS_CODES) == 65
    assert len(set(CLASS_CODES.values())) == 65  # 符号が重ならない


def test_labels_of_test_videos_map_to_classes():
    assert class_for_label("」") == "。"
    assert class_for_label("（") == "（" and code_for_label("）") == ".-..-."
    assert class_for_label("A") is None


def test_units_for_char():
    assert units_for_char("（") == ["-.--.-"]
    assert units_for_char("ガ") == [".-..", ".."]
