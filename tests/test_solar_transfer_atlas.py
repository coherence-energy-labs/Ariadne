from ariadne.interplanetary.solar_atlas import (
    AtlasBody,
    TransferCorridor,
    shortest_corridor_route,
)


def test_shortest_corridor_route_prefers_multileg_when_score_is_lower():
    corridors = [
        TransferCorridor("EARTH", "SATURN", 0.0, 1.0, 1.0, 9000.0, 20.0, 4.0, 5.0, 20.0),
        TransferCorridor("EARTH", "JUPITER", 0.0, 1.0, 1.0, 3000.0, 10.0, 3.0, 4.0, 4.0),
        TransferCorridor("JUPITER", "SATURN", 0.0, 1.0, 1.0, 3000.0, 10.0, 3.0, 4.0, 5.0),
    ]
    path, score = shortest_corridor_route(corridors, "EARTH", "SATURN")
    assert path == ("EARTH", "JUPITER", "SATURN")
    assert score == 9.0


def test_public_api_exports_solar_transfer_atlas_builder():
    import ariadne

    assert callable(ariadne.build_solar_transfer_atlas)


def test_atlas_body_records_ephemeris_identity():
    body = AtlasBody("EARTH", "EARTH", 1.0, "#3f7fd2")
    assert body.name == "EARTH"
    assert body.ephemeris_name == "EARTH"

