"""App tổng hợp kiểu CRUD cho dữ liệu SFT (roadmap đề xuất 5, proposal 5.5).

    domains.py    đặc tả app: collection, trường, tool, mẫu câu tiếng Việt
    apps.py       dựng một app (tên ngẫu nhiên, catalog, state) và chạy tool thật trên lớp App của zaic_tools
    scenarios.py  sinh lời nói, lịch sử và quỹ đạo gold; `generate(hosts, n, seed)`
    vi.py         số, tiền, giờ, ngày nói bằng tiếng Việt
"""
from .apps import SynthEnv
from .scenarios import generate, replay, specs, transcript

__all__ = ["SynthEnv", "generate", "replay", "specs", "transcript"]
