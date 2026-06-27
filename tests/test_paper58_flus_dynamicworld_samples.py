from scripts.paper58_benchmark.fetch_dynamicworld_flus_samples import (
    band_names_for_years,
    matches_admin_filter,
    parse_admin_filter,
)


def test_parse_admin_filter_accepts_province_city_county() -> None:
    parsed = parse_admin_filter("广东省|东莞市|东莞市")

    assert parsed == {"province": "广东省", "city": "东莞市", "county": "东莞市"}


def test_matches_admin_filter_uses_only_supplied_fields() -> None:
    properties = {"省": "广东省", "市": "东莞市", "县": "东莞市", "乡": "虎门镇"}

    assert matches_admin_filter(properties, {"province": "广东省", "city": "东莞市"})
    assert not matches_admin_filter(properties, {"province": "江苏省", "city": "东莞市"})


def test_band_names_for_years_include_labels_and_scaled_probabilities() -> None:
    names = band_names_for_years([2020, 2021])

    assert names[:2] == ["flus_label_2020", "flus_label_2021"]
    assert "p_construction_2020" in names
    assert "p_arable_2021" in names
    assert len(names) == 14
