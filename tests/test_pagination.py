"""懒加载分页（offset/has_more）契约测试。

测的是公开行为：同一查询、不同 offset 的切片正确性与 has_more 边界，
以及默认请求（不带 offset）与旧行为逐位一致（向后兼容）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_api_module():
    """以真实模块路径加载 05_api（与其他测试同模式）。"""
    spec = importlib.util.spec_from_file_location(
        "api_05", REPO / "scripts" / "05_api.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- paginate_results 纯函数（切片语义） ----------


class TestPaginateResults:
    def setup_method(self):
        self.mod = _load_api_module()
        # 构造 45 条假结果
        self.results = [{"rank": i + 1, "video_id": f"v{i}"} for i in range(45)]

    def test_first_page(self):
        window, has_more = self.mod.paginate_results(self.results, 0, 20)
        assert len(window) == 20
        assert window[0]["rank"] == 1
        assert window[-1]["rank"] == 20
        assert has_more is True

    def test_second_page(self):
        window, has_more = self.mod.paginate_results(self.results, 20, 20)
        assert len(window) == 20
        assert window[0]["rank"] == 21
        assert window[-1]["rank"] == 40
        assert has_more is True

    def test_tail_page_short(self):
        window, has_more = self.mod.paginate_results(self.results, 40, 20)
        assert len(window) == 5
        assert window[-1]["rank"] == 45
        assert has_more is False

    def test_offset_beyond_end(self):
        window, has_more = self.mod.paginate_results(self.results, 100, 20)
        assert window == []
        assert has_more is False

    def test_exact_fit(self):
        """结果数恰好等于 offset+top_k：无更多。"""
        window, has_more = self.mod.paginate_results(self.results, 25, 20)
        assert len(window) == 20
        assert has_more is False

    def test_fewer_than_page(self):
        small = self.results[:5]
        window, has_more = self.mod.paginate_results(small, 0, 20)
        assert len(window) == 5
        assert has_more is False
