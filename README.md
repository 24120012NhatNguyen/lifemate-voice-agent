# ZAIC 2026 · LifeMate

Giải pháp của đội cho bảng **LifeMate** của Zalo AI Challenge 2026: agent nhận lệnh tiếng Việt bằng giọng nói và gọi tool trên môi trường `zaic_tools`, chấm bằng `state_hash` và F1 trên danh sách call.

## Tài liệu

- [docs/PROPOSAL.md](docs/PROPOSAL.md): đề xuất kiến trúc pipeline, đang chờ duyệt.

## Dữ liệu

Dữ liệu thi không nằm trong repo. Đặt gói dữ liệu của BTC cạnh repo:

```
zalo-ai-2026/
├── data/lifemate/
│   ├── train/          train.jsonl, audio/, train_environment/, zaic_tools/
│   └── public_test/    public_test.jsonl, audio/, public_test_environment/, zaic_tools/
└── lifemate/           repo này
```

Không sửa `zaic_tools/` hay `*_environment/`: BTC chấm bằng bản gốc.

## Ràng buộc chính

- Tổng tham số dưới 6 tỷ, LLM dưới 5 tỷ, ngữ cảnh 16K.
- 30 giây mỗi mẫu trên một GPU A30, chạy offline trong Docker.
- Mỗi mẫu đúng một `ToolEnv`; mọi call đều tính vào bài nộp.

## Quy ước làm việc

- Commit theo [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Mỗi thay đổi đi qua pull request vào `main`.
- Không commit dữ liệu thi, trọng số mô hình hay file nộp.
