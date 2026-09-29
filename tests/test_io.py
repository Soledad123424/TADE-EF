from pathlib import Path

from tade_ef.io import parse_raw_header, read_csv, write_csv


def test_parse_evt3_header(tmp_path: Path) -> None:
    raw = tmp_path / "sample.raw"
    raw.write_bytes(b"% format EVT3; height = 720; width = 1280\n% end\n\x00\x00")
    header = parse_raw_header(raw)
    assert header.event_format == "EVT3"
    assert (header.width, header.height) == (1280, 720)


def test_write_csv_uses_union_of_row_fields(tmp_path: Path) -> None:
    output = tmp_path / "rows.csv"
    write_csv(output, [{"a": 1}, {"a": 2, "b": 3}])
    rows = read_csv(output)
    assert rows[0] == {"a": "1", "b": ""}
    assert rows[1] == {"a": "2", "b": "3"}
