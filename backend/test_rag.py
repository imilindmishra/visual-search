from rag import get_base_id, parse_query, rrf_fuse


def test_get_base_id_strips_view_suffix():
    """Numeric/angle suffixes after '<...>_id_<digits>' are stripped."""
    assert get_base_id("MEN_Denim_id_00005724_01_3_back") == "MEN_Denim_id_00005724"


def test_get_base_id_no_id_token_returns_unchanged():
    """IDs without an 'id' token pass through unchanged."""
    assert get_base_id("SOME_RANDOM_PRODUCT_42") == "SOME_RANDOM_PRODUCT_42"


def test_parse_query_extracts_color_category_gender():
    """Free text resolves to color/category/gender, mapped onto real Qdrant categories."""
    assert parse_query("blue tshirt men") == {"color": "blue", "category": "sweatshirts", "gender": "men"}


def test_parse_query_polo_maps_to_real_category():
    """'polo' must map to an existing category2 value — there is no 'polos' bucket in the data."""
    parsed = parse_query("red polo")
    assert parsed["category"] == "shirts"


def test_parse_query_women_overrides_men_when_both_present():
    """'women' takes precedence — matches the gender-detection order in parse_query."""
    assert parse_query("men and women jackets")["gender"] == "women"


def test_parse_query_no_matches_returns_all_none():
    """A query with no recognizable color/category/gender yields an all-None dict."""
    assert parse_query("something completely unrelated") == {"color": None, "category": None, "gender": None}


def test_rrf_fuse_combines_rankings_by_reciprocal_rank():
    """Items appearing earlier and in more rankings score higher."""
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == fused["b"]  # both rank 0 once and rank 1 once
    assert fused["a"] > fused["c"]   # 'a' appears in both rankings, 'c' only one
    assert fused["c"] > 0 and fused["d"] > 0


def test_rrf_fuse_empty_rankings_returns_empty():
    """No rankings means no scores."""
    assert rrf_fuse([]) == {}
