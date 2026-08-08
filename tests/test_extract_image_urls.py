# ============================================================
# extract_image_urls 单元测试
# 覆盖：JSON 字符串解析 / 去重 / 空值与缺失 / 解析异常兜底
# ============================================================

from src.knowledge.doc_rag import extract_image_urls


def test_extract_from_json_string():
    hits = [{"image_urls": '["http://a/1.png", "http://a/2.png"]'}]
    assert extract_image_urls(hits) == ["http://a/1.png", "http://a/2.png"]


def test_dedup_across_hits():
    hits = [
        {"image_urls": '["http://a/1.png"]'},
        {"image_urls": '["http://a/1.png", "http://a/3.png"]'},
    ]
    assert extract_image_urls(hits) == ["http://a/1.png", "http://a/3.png"]


def test_empty_and_missing():
    assert extract_image_urls([]) == []
    assert extract_image_urls([{"image_urls": ""}]) == []
    assert extract_image_urls([{}]) == []


def test_invalid_json_skipped():
    hits = [{"image_urls": "not-json"}]
    assert extract_image_urls(hits) == []
