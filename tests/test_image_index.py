"""06b_index_images.py 图片索引管线的单元测试。

只使用内存假库/假 encoder/假文件树，不读取 NAS、真实向量库或真实模型。
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "06b_index_images.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("index_images_test", str(SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_06():
    spec = importlib.util.spec_from_file_location(
        "index_images_test_06", str(SCRIPTS / "06_index_all.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- 1. 扩展名过滤 ----------

def test_discover_filters_extensions(tmp_path):
    """jpg/jpeg/png/webp（含大写扩展名）入，psd/ai/tif/heic/其余不出。"""
    module = _load_module()
    root = tmp_path / "mount"
    root.mkdir()
    keep = ["a.jpg", "b.jpeg", "c.png", "d.webp", "e.JPG", "f.PNG", "g.WEBP"]
    drop = ["x.psd", "y.ai", "z.tif", "w.heic", "v.pdf", "u.gif", "t.bmp", "s.mp4"]
    for name in keep + drop:
        (root / name).write_bytes(b"raw")
    # 隐藏文件不入库
    (root / ".hidden.jpg").write_bytes(b"raw")

    found = module.discover_images([str(root)], include_dir=[], exclude_dir=[],
                                   min_size_mb=0.0)
    names = sorted(Path(item["image_path"]).name for item in found)
    assert names == sorted(keep)


def test_discover_requires_min_size(tmp_path):
    """min_size_mb 门槛生效（默认跳过 0 字节文件）。"""
    module = _load_module()
    root = tmp_path / "mount"
    root.mkdir()
    big = root / "big.jpg"
    big.write_bytes(b"x" * 2048)
    (root / "tiny.jpg").write_bytes(b"")  # 0 字节

    found = module.discover_images([str(root)], include_dir=[], exclude_dir=[],
                                   min_size_mb=0.0)
    names = [Path(item["image_path"]).name for item in found]
    assert names == ["big.jpg"]


# ---------- 2. 目录剪枝 ----------

def test_discover_prunes_recycle_snapshot_hidden(tmp_path):
    """#recycle/#snapshot/@eadir/隐藏段/.fcpbundle/代码目录全部剪掉。"""
    module = _load_module()
    root = tmp_path / "mount"
    root.mkdir()
    good = root / "keep"
    good.mkdir()
    (good / "ok.png").write_bytes(b"raw")
    for d in ("#recycle", "#snapshot", "@eadir", ".hidden_dir", "x.fcpbundle",
              "site-packages", "node_modules"):
        sub = root / d
        sub.mkdir()
        (sub / "bad.jpg").write_bytes(b"raw")
    # 隐藏文件也不入
    (root / ".tmp.png").write_bytes(b"raw")

    found = module.discover_images([str(root)], include_dir=[], exclude_dir=[],
                                   min_size_mb=0.0)
    names = [Path(item["image_path"]).name for item in found]
    assert names == ["ok.png"]


def test_discover_respects_include_exclude_dir(tmp_path):
    """include/exclude 子串匹配与 06 语义一致。"""
    module = _load_module()
    root = tmp_path / "mount"
    root.mkdir()
    (root / "alpha").mkdir()
    (root / "beta").mkdir()
    (root / "alpha" / "in.jpg").write_bytes(b"raw")
    (root / "beta" / "out.jpg").write_bytes(b"raw")

    found = module.discover_images([str(root)], include_dir=["alpha"],
                                   exclude_dir=[], min_size_mb=0.0)
    assert [Path(i["image_path"]).name for i in found] == ["in.jpg"]

    found = module.discover_images([str(root)], include_dir=[],
                                   exclude_dir=["beta"], min_size_mb=0.0)
    assert [Path(i["image_path"]).name for i in found] == ["in.jpg"]


# ---------- 3. image_key 同构性（独立期望值，写死字面量） ----------

def test_image_key_matches_video_key_construction():
    """md5(f"{share}/{relpath}")[:8] 与手算字面量一致，且与 06 的 video_key_of 同构。"""
    module = _load_module()
    v06 = _load_06()
    cases = [
        ("share_a", "dir1/photo.jpg"),
        ("share_b", "nested/deep/IMG_0001.PNG"),
        ("share_c", "2024年拍摄/成品/最终版.webp"),
        ("share_d", "plain.png"),
        ("share_e", "中文目录/图 片.jpeg"),  # 中文与空格
    ]
    for share, relpath in cases:
        expected = hashlib.md5(f"{share}/{relpath}".encode("utf-8")).hexdigest()[:8]
        assert module.image_key_of(share, relpath) == expected
        assert module.image_key_of(share, relpath) == v06.video_key_of(share, relpath)


# ---------- 4. state 幂等 ----------

def test_state_idempotent_skip_and_mtime_change(tmp_path):
    """mtime/size 都没变 → 计划阶段跳过；mtime 变化 → 重新进入待处理。"""
    module = _load_module()
    root = tmp_path / "mount"
    root.mkdir()
    p_old = root / "old.jpg"
    p_new = root / "new.jpg"
    p_old.write_bytes(b"old")
    p_new.write_bytes(b"new")

    found = module.discover_images([str(root)], include_dir=[], exclude_dir=[],
                                   min_size_mb=0.0)
    items = {Path(i["image_path"]).name: i for i in found}
    key_old = module.image_key_of(items["old.jpg"]["share"],
                                  items["old.jpg"]["relpath"])
    key_new = module.image_key_of(items["new.jpg"]["share"],
                                  items["new.jpg"]["relpath"])

    state = {
        key_old: {"status": "done", "mtime": items["old.jpg"]["mtime"],
                  "size": items["old.jpg"]["size"],
                  "frame_path": items["old.jpg"]["image_path"]},
    }
    # 未变 -> 跳过
    plan = module.plan_pending(found, state, force=False)
    assert [i["image_key"] for i in plan] == [key_new]

    # mtime 变了 -> 重处理
    state[key_old]["mtime"] = items["old.jpg"]["mtime"] + 123.0
    plan = module.plan_pending(found, state, force=False)
    assert sorted(i["image_key"] for i in plan) == sorted([key_old, key_new])

    # failed 状态可重试
    state[key_old] = {"status": "failed", "mtime": items["old.jpg"]["mtime"],
                      "size": items["old.jpg"]["size"], "error": "x"}
    plan = module.plan_pending(found, state, force=False)
    assert sorted(i["image_key"] for i in plan) == sorted([key_old, key_new])

    # force 全部重跑
    state[key_old] = {"status": "done", "mtime": items["old.jpg"]["mtime"],
                      "size": items["old.jpg"]["size"],
                      "frame_path": items["old.jpg"]["image_path"]}
    plan = module.plan_pending(found, state, force=True)
    assert sorted(i["image_key"] for i in plan) == sorted([key_old, key_new])


def test_state_roundtrip_atomic(tmp_path):
    """save_state 走 tmp+rename，load_state 读回一致；损坏文件不炸。"""
    module = _load_module()
    db = tmp_path / "db"
    state = {"abcd1234": {"status": "done", "mtime": 1.0, "size": 2,
                          "frame_path": "/x/y.jpg"}}
    module.save_state(db, state)
    assert not (db / "state_images.json.tmp").exists()
    assert module.load_state(db) == state

    # 损坏文件 -> 从空开始
    p = db / "state_images.json"
    p.write_text("{not json", encoding="utf-8")
    assert module.load_state(db) == {}

    # 显式 .json 文件路径也支持（CLI --state 传文件而非目录）
    f = tmp_path / "custom" / "my_state.json"
    module.save_state(f, state)
    assert f.exists()
    assert module.load_state(f) == state
    assert not f.with_suffix(".json.tmp").exists()


# ---------- 5. metadata 契约（假 encoder + 假 collection） ----------

class _FakeCollection:
    """记录 upsert 调用的内存假 collection。"""

    def __init__(self):
        self.upserts: list[dict] = []

    def upsert(self, ids, embeddings, metadatas, documents):
        self.upserts.append({"ids": list(ids), "embeddings": embeddings,
                             "metadatas": metadatas, "documents": documents})


class _FakeEncoder:
    name = "fake"
    model_id = "fake/model"
    dim = 4

    def encode_images(self, imgs):
        return [[0.1, 0.2, 0.3, 0.4] for _ in imgs]


def _make_image(tmp_path, name="photo.jpg", data=b"fake", max_side=None):
    """造一张真 PNG/JPG 测试图：默认 32x24 纯色。"""
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    img = Image.new("RGB", (32, 24), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


def test_process_image_metadata_contract(tmp_path):
    """单图处理路径：upsert 收到的 id 格式、metadata 字段与取值全对齐契约。"""
    module = _load_module()
    pytest.importorskip("PIL")
    img_path = _make_image(tmp_path, "photo.jpg")
    item = {
        "image_path": str(img_path),
        "share": "share_x",
        "relpath": "dir/photo.jpg",
        "stem": "photo",
        "size": img_path.stat().st_size,
        "mtime": img_path.stat().st_mtime,
    }
    item["image_key"] = module.image_key_of(item["share"], item["relpath"])

    col = _FakeCollection()
    result = module.process_image(item, _FakeEncoder(), col, max_side=1024)

    assert result["ok"] is True
    assert len(col.upserts) == 1
    up = col.upserts[0]
    assert up["ids"] == [f"{item['image_key']}_photo@0.0"]

    meta = up["metadatas"][0]
    assert meta["image_key"] == item["image_key"]
    assert meta["video_id"] == "photo"
    assert meta["video_name"] == "photo.jpg"
    assert meta["media_type"] == "image"
    assert meta["time"] == 0.0
    assert meta["image_path"] == str(img_path)
    assert meta["frame_path"] == str(img_path)
    assert meta["share"] == "share_x"
    assert meta["relpath"] == "dir/photo.jpg"
    assert meta["width"] == 32
    assert meta["height"] == 24
    assert set(meta) == {
        "image_key", "video_id", "video_name", "media_type", "time",
        "image_path", "width", "height", "share", "relpath", "frame_path",
    }
    # document 含文件名便于检索层调试
    assert "photo" in up["documents"][0]
    # 返回 state 条目
    assert result["state"]["status"] == "done"
    assert result["state"]["frame_path"] == str(img_path)
    assert result["state"]["mtime"] == item["mtime"]
    assert result["state"]["size"] == item["size"]


def test_process_image_downscales_large_side(tmp_path):
    """最长边 >1024 时等比缩到 1024（metadata 记原始宽高）。"""
    module = _load_module()
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    img = Image.new("RGB", (2048, 1024), (10, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_path = tmp_path / "big.png"
    img_path.write_bytes(buf.getvalue())
    item = {
        "image_path": str(img_path),
        "share": "share_x",
        "relpath": "dir/big.png",
        "stem": "big",
        "size": img_path.stat().st_size,
        "mtime": img_path.stat().st_mtime,
    }
    item["image_key"] = module.image_key_of(item["share"], item["relpath"])

    seen: list[list] = []

    class _Spy(_FakeEncoder):
        def encode_images(self, imgs):
            seen.append([im.size for im in imgs])
            return super().encode_images(imgs)

    col = _FakeCollection()
    result = module.process_image(item, _Spy(), col, max_side=1024)
    assert result["ok"] is True
    # 缩放后尺寸：最长边 1024，等比
    assert seen[0][0] == (1024, 512)
    meta = col.upserts[0]["metadatas"][0]
    assert meta["width"] == 2048
    assert meta["height"] == 1024


# ---------- 5b. 网页缩略图（thumb_path） ----------

def test_thumbnail_generated_bounded_and_webp(tmp_path):
    """缩略图：最长边限到 512、输出 WebP、保持宽高比、写入 {image_key}.webp。"""
    module = _load_module()
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    img = Image.new("RGB", (2048, 1024), (10, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_path = tmp_path / "wide.jpg"
    img_path.write_bytes(buf.getvalue())
    item = {
        "image_path": str(img_path), "share": "s", "relpath": "d/wide.jpg",
        "stem": "wide", "size": img_path.stat().st_size,
        "mtime": img_path.stat().st_mtime,
    }
    item["image_key"] = module.image_key_of(item["share"], item["relpath"])
    thumbs = tmp_path / "thumbs"

    col = _FakeCollection()
    result = module.process_image(item, _FakeEncoder(), col, max_side=1024,
                                  thumbs_dir=thumbs)
    assert result["ok"] is True

    meta = col.upserts[0]["metadatas"][0]
    assert meta["thumb_path"] == str(thumbs / f"{item['image_key']}.webp")
    # frame_path 仍指原图（契约不变）
    assert meta["frame_path"] == str(img_path)

    from PIL import Image as PILImage
    with PILImage.open(meta["thumb_path"]) as t:
        assert t.format == "WEBP"
        assert max(t.size) == 512
        # 2048x1024 等比缩到 512x256（宽高比保持）
        assert t.size == (512, 256)


def test_thumbnail_idempotent_skips_when_not_older(tmp_path):
    """缩略图已存在且不早于原图 mtime -> 不重写（mtime 保持不变）。"""
    module = _load_module()
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    img = Image.new("RGB", (800, 600), (1, 2, 3))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_path = tmp_path / "p.jpg"
    img_path.write_bytes(buf.getvalue())
    item = {
        "image_path": str(img_path), "share": "s", "relpath": "d/p.jpg",
        "stem": "p", "size": img_path.stat().st_size,
        "mtime": img_path.stat().st_mtime,
    }
    item["image_key"] = module.image_key_of(item["share"], item["relpath"])
    thumbs = tmp_path / "thumbs"

    p1 = module.save_thumbnail(
        module.open_image(str(img_path))[0], thumbs, item["image_key"],
        item["mtime"])
    first_mtime = Path(p1).stat().st_mtime_ns
    # 再存一次：同 mtime 下应判定"不旧于原图"直接返回，不重写文件
    p2 = module.save_thumbnail(
        module.open_image(str(img_path))[0], thumbs, item["image_key"],
        item["mtime"])
    assert p1 == p2
    assert Path(p2).stat().st_mtime_ns == first_mtime


def test_thumbnail_write_failure_does_not_break_index(tmp_path, monkeypatch):
    """缩略图写盘失败 -> 本张仍入库（thumb_path 缺省），不中断。"""
    module = _load_module()
    pytest.importorskip("PIL")

    img_path = _make_image(tmp_path, "p.jpg")
    item = {
        "image_path": str(img_path), "share": "s", "relpath": "d/p.jpg",
        "stem": "p", "size": img_path.stat().st_size,
        "mtime": img_path.stat().st_mtime,
    }
    item["image_key"] = module.image_key_of(item["share"], item["relpath"])

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(module, "save_thumbnail", boom)
    col = _FakeCollection()
    result = module.process_image(item, _FakeEncoder(), col, max_side=1024,
                                  thumbs_dir=tmp_path / "thumbs")

    assert result["ok"] is True
    meta = col.upserts[0]["metadatas"][0]
    assert "thumb_path" not in meta
    assert meta["frame_path"] == str(img_path)


# ---------- 6. 熔断 ----------

def test_fuse_after_consecutive_failures():
    """连续失败计数达到阈值 -> 熔断标志置位并停止处理。"""
    module = _load_module()
    items = [{"image_key": f"k{i:03d}", "image_path": f"/x/{i}.jpg"} for i in range(30)]

    def always_fail(item, encoder, col, max_side=1024):
        return {"ok": False, "error": "boom", "state": None}

    plan = module.run_processing(
        items, encoder=_FakeEncoder(), collection=_FakeCollection(),
        process_one=always_fail, batch_size=64, max_consecutive_failures=20,
        save=None, disk_check=lambda: None,
    )
    # 30 个全部失败，熔断在第 20 个后触发：不再处理后续项
    assert plan["processed"] == 20
    assert plan["failed"] == 20
    assert plan["fused"] is True

    # 成功即清零：前 25 成功 + 后 5 失败不熔断
    calls: list[dict] = []

    def fail_at_end(item, encoder, col, max_side=1024):
        calls.append(item)
        if len(calls) > 25:
            return {"ok": False, "error": "boom", "state": None}
        return {"ok": True, "state": {"status": "done"}, "meta": {}}

    plan = module.run_processing(
        items, encoder=_FakeEncoder(), collection=_FakeCollection(),
        process_one=fail_at_end, batch_size=64, max_consecutive_failures=20,
        save=None, disk_check=lambda: None,
    )
    assert plan["fused"] is False
    assert plan["processed"] == 30


def test_disk_guard_stops_before_low_space():
    """可用空间 < 阈值 -> 停机（返回保护标志），不再处理。"""
    module = _load_module()
    items = [{"image_key": f"k{i:03d}", "image_path": f"/x/{i}.jpg"} for i in range(10)]
    calls: list = []

    def ok(item, encoder, col, max_side=1024):
        calls.append(item)
        return {"ok": True, "state": {"status": "done"}, "meta": {}}

    plan = module.run_processing(
        items[:1], encoder=_FakeEncoder(), collection=_FakeCollection(),
        process_one=ok, batch_size=64, max_consecutive_failures=20,
        save=None, disk_check=lambda: (_ for _ in ()).throw(
            module.LowDiskSpace("free 1.0GB < 5.0GB")),
    )
    assert plan["processed"] == 0
    assert plan["stopped"] is True
    assert calls == []
