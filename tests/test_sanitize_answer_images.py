# ============================================================
# _sanitize_answer_images 单元测试
# 覆盖：占位符 URL（含省略号）剥成描述 / 裸占位 URL 删除 /
#       valid_urls 为空时全部剥掉 / 真实 URL 保留 / 无图不处理
# ============================================================

from src.knowledge.fusion import _sanitize_answer_images as S

REAL = "http://minio:9000/knowledge-docs/images/doc123/table/p4_189_103.png"


def test_placeholder_image_stripped_to_alt():
    out = S("前文 ![表格原图](http://...url.../) 后文", {REAL})
    assert out == "前文 表格原图 后文"


def test_placeholder_uppercase_stripped_even_when_no_valid():
    out = S("前文 ![表格原图](http://...URL.../) 后文", set())
    assert out == "前文 表格原图 后文"


def test_bare_placeholder_url_removed():
    out = S("裸地址 http://...url.../ 文本", set())
    assert out == "裸地址 文本"


def test_real_url_kept_when_valid():
    out = S(f"真实图 ![表格原图]({REAL}) 保留", {REAL})
    assert out == f"真实图 ![表格原图]({REAL}) 保留"


def test_fake_url_stripped_when_not_valid():
    out = S(f"编造图 ![表格原图]({REAL}) 不该留", set())
    assert out == "编造图 表格原图 不该留"


def test_no_images_unchanged():
    assert S("无图纯文本", set()) == "无图纯文本"
