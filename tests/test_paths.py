from mobile_network_sim.paths import cache_root


def test_cache_root_uses_user_writable_location(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    expected = (tmp_path / "mobile-network-sim").resolve()
    assert cache_root() == expected
    assert expected.is_dir()