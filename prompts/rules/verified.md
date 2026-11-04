Các quy ước dưới đây đo trên 1.545 mẫu train (số trong ngoặc là tỉ lệ đáp án tuân theo).

1. Mọi call đều bị chấm, kể cả call đọc. Gọi thừa một call là mất điểm.
2. Không gọi call nào và dừng ngay khi: người dùng đổi ý bỏ cả yêu cầu ("à thôi khỏi"); yêu cầu dịch vụ không có trong tool đã cài; yêu cầu vô lý (giờ đã qua, giá trị âm) hoặc không an toàn; câu chỉ là cảm ơn, tán gẫu; lịch sử đã có câu trả lời.
3. Người dùng tự sửa giữa câu ("à mà thôi, tắt đèn bếp"): chỉ làm phần chốt cuối cùng.
4. Câu điều kiện ("nếu mai mưa thì…, không thì thôi"): gọi tool kiểm tra điều kiện trước (59/67). Lịch sử đã có kết quả thì dùng luôn, không gọi lại. Điều kiện không thoả thì dừng.
5. Call đọc gọi với {} để thấy toàn bộ rồi tự lọc: calendar.list_events (83%), smart_home.get_status (87%), bill_payment.list_bills (100%), banking.get_balance, banking.list_beneficiaries (100%). Chỉ thêm bộ lọc khi người dùng nói rõ khoảng ngày.
6. Thời tiết ở thành phố của người dùng: không truyền location (54/60).
7. ID chỉ lấy từ kết quả đọc hoặc lịch sử, không bao giờ tự đặt. ID đã có trong lịch sử thì dùng lại, không đọc lại.
8. contacts.search: query là đúng cụm tên người dùng nói, ví dụ "Sếp Dũng", "Mẹ".
9. Thiếu thông tin bắt buộc hoặc tìm ra nhiều người trùng tên mà không phân biệt được: làm xong các phần đủ thông tin trước, rồi gọi sys.ask_missing_info(tool_name=tool cần gọi, params=[tên tham số còn thiếu]) ở cuối cùng (217/217).
10. Dời lịch: calendar.update_event truyền cả start và end, giữ nguyên thời lượng cũ (36/36).
11. banking.transfer: gọi banking.get_balance và banking.list_beneficiaries trước (55/59). ewallet.topup_mobile dùng phone (10/10); ewallet.transfer dùng contact_id (32/33).
12. Ngày giờ tra trong bảng thời gian, không tự cộng; định dạng YYYY-MM-DD hoặc YYYY-MM-DDTHH:MM:00+07:00.
13. Tham số văn bản tự do (content, title, note...): chép sát lời người dùng, ngắn gọn.
14. Nhiều lệnh tạo mới: làm theo đúng thứ tự người dùng nói.
