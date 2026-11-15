# Luật nhẹ cho bản cuối (Pro, 15/11/2026)

Nhánh `pro/light-rules` (worktree `fable-work/pro-lightrules`, từ `codex/stop-gate` 07e01a4): commit f5f5eeb + eff0dc4.
Mọi cờ nằm trong `guards:`, mặc định TẮT; test `tests/test_light_rules.py` (13 test) chứng minh tắt cờ = hành vi cũ và phát
lại gold train với cờ bật không đổi state, điểm FAQ 1.0. Toàn bộ test: 381 đạt. Không dùng public test, tên app, ID hay schema
riêng của public; mỗi luật dựa trên định dạng chung của catalog (`x-ref`, mô tả "mặc định", "YYYY-MM-DD") và FAQ.

## Bốn luật

| Cờ | Làm gì | Căn cứ train/val |
|---|---|---|
| `xref_guard` | Tham số có `x-ref` "app.C" nhận một ID chỉ thấy trong một bảng x-ref khác (vd. `course_id=evt_…`, `contact_id=chat_…`): không chạy, trả `done:false` + ghi chú tool nguồn | Gold train: 0 / 1.725 call bị chặn. Vết 260/255/240: 2 va + 1 vr mẫu mỗi lượt (1435, 1576, 1348, 1166) |
| `explicit_defaults` | Tool ĐỌC thiếu tham số tuỳ chọn mà mô tả nêu mặc định không cần state ("bỏ trống = thành phố của người dùng", "mặc định motorbike" trong enum): điền giá trị đó | Kết quả tool không đổi (state thời tiết luôn chỉ có thành phố hồ sơ, 634/634; maps dùng motorbike khi trống). FAQ câu 5: tham số tuỳ chọn thừa vẫn khớp. Không bao giờ đụng tool ghi |
| `ask_schema` | LLM hỏi lại tham số không có trong schema: bỏ tham số đó; không còn gì thì không ghi câu hỏi, trả ghi chú tham số bắt buộc (một lần mỗi tool) | Gold train 0/217 câu hỏi có tham số ngoài schema; FAQ câu 6 (sai tên tham số = không khớp). RESEARCH_75 R3 |
| `day_filter` | Tool ĐỌC có tham số ngày YYYY-MM-DD (không giờ) bỏ trống hết, lời lượt cuối nêu đúng 1 ngày: điền ngày đó | Gold train khi lời nêu đúng 1 ngày: 23/23 call có ngày trùng ngày đó; 89 call bỏ trống mà lọc theo ngày đó không giấu ID nào gold dùng sau (89/89) |

## Số đo

Offline: phát lại vết 260 LM-91 (dạng public, soup 211, 2 lượt), chấm FAQ. Chỉ đo được phần đổi call; phản ứng của LLM với
ghi chú (xref_guard, ask_schema) chỉ đo được trên GPU.

| Luật | val_app r1 / r2 | val_random r1 / r2 | Tốt / tệ |
|---|---|---|---|
| xref_guard | 0,00 / 0,00 | 0,00 / 0,00 | 0/0 (mẫu kích hoạt đều đang 0 điểm vì hash) |
| explicit_defaults | +0,17 / +0,17 | +0,15 / +0,15 | 1/0 mỗi tập (1534, 0037) |
| ask_schema | 0,00 / 0,00 | 0,00 / 0,00 | 0/0 (0828 cần LLM gọi lại) |
| day_filter | +0,19 / +0,19 | +0,30 / +0,30 | 2/0 va (1038, 1158), 1/0 vr (1180) |
| **Cả bốn** | **+0,36 / +0,36** | **+0,44 / +0,44** | **3/0 va, 2/0 vr** |

Trên vết 255 (bản chép đúng) và 240, day_filter còn cứu 0808, 1013, 0690, 1213 (+0,3…+0,7 điểm).

**GPU 267** (`configs/experiments/267_light_pub{,_val-app}.yaml` = 260_base_pub + 4 cờ; 2 lượt lặp; script
`tools-pro/lm267_run.sh`, đang xếp hàng khoá GPU lúc 18:45 sau 268 và 265): kết quả tự ghi vào night.log dòng
"KẾT QUẢ 267" và `fable-work/pro-lightrules/reports/lm267/{val_app,val_random}-{faq,strict}.txt` (arm_compare `--light`, có nhóm
`rule:<cờ>` cho từng luật và `light_untriggered` để đọc nhiễu). CHƯA có số GPU khi viết báo cáo này.

## Khuyến nghị cho bản cuối

- **Bật `explicit_defaults` và `day_filter`** ngay cả khi chưa có số GPU: lợi offline dương ở cả hai tập, 0 mẫu tệ, kết quả tool
  không đổi (explicit_defaults) hoặc đã kiểm không giấu ID trên toàn gold train (day_filter).
- **`xref_guard`, `ask_schema`: bật nếu 267 cho nhóm `rule:` tương ứng không âm ở cả hai tập**; nếu âm thì tắt. Lợi kỳ vọng nhỏ
  (≤ +0,3), rủi ro thấp (gold 0 chặn nhầm).
- Tiêu chí nhận như nhiệm vụ: mẫu kích hoạt lợi ở cả hai tập và không tập nào giảm. Lưu ý chấm 267 bằng FAQ (strict sẽ báo lỗ giả
  cho explicit_defaults/day_filter vì tham số thừa).

## Đã xem và bỏ

ask_resolve dùng danh sách lượt trước (1 kích hoạt/lượt, đúng một phần); chặn call đọc trùng (≈0); chạy thử call ghi trên bản sao
môi trường để bỏ call `done:false` (lợi trực tiếp +0,02, và là lấy thông tin ngoài API tool: vùng xám, không làm); đổi tên chương
trình thành ID (chỉ một app giữ lại ở val, không có ở public: không chuyển được).
