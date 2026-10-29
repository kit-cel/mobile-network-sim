from mobile_network_sim.cli import main


def test_radio_rrc_cli_writes_summary(tmp_path) -> None:
    output = tmp_path / "summary.json"
    exit_code = main(
        [
            "radio-rrc",
            "--config",
            "examples/radio_rrc.yaml",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert output.is_file()
    assert '"num_steps": 20' in output.read_text(encoding="utf-8")
