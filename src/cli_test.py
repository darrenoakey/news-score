import pytest

from src.cli import format_response, main


# ##################################################################
# test format response
# verifies json formatting produces indented output
def test_format_response():
    data = {"url": "https://example.com", "rank": 7.5, "source": "cache"}
    result = format_response(data)
    assert '"url": "https://example.com"' in result
    assert '"rank": 7.5' in result
    assert '"source": "cache"' in result


# ##################################################################
# test format response empty
# verifies empty dict formats correctly
def test_format_response_empty():
    result = format_response({})
    assert result == "{}"


# ##################################################################
# test main missing command
# verifies missing command causes error
def test_main_missing_command():
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2


# ##################################################################
# test main rank missing url
# verifies rank command requires url argument
def test_main_rank_missing_url():
    with pytest.raises(SystemExit) as exc_info:
        main(["rank"])
    assert exc_info.value.code == 2


# ##################################################################
# test main train missing args
# verifies train command requires both url and score
def test_main_train_missing_args():
    with pytest.raises(SystemExit) as exc_info:
        main(["train"])
    assert exc_info.value.code == 2


# ##################################################################
# test main train missing score
# verifies train command requires score argument
def test_main_train_missing_score():
    with pytest.raises(SystemExit) as exc_info:
        main(["train", "https://example.com"])
    assert exc_info.value.code == 2


# ##################################################################
# test main train invalid score
# verifies train command validates score is a number
def test_main_train_invalid_score():
    with pytest.raises(SystemExit) as exc_info:
        main(["train", "https://example.com", "not_a_number"])
    assert exc_info.value.code == 2
