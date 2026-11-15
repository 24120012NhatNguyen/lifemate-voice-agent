"""Synth v3: app tổng hợp viết theo đúng phong cách catalog train, chạy THẬT bên trong ToolEnv (Session của zaic_tools).

Khác synth v1/v2 (argon-lm85-code-audit.md, phần 2):
- Không chỉ có khuôn phẳng `<sing>_id` + x-ref: có app trạng thái thiết bị không ID (water_heater), tham số là TÊN phải
  lấy từ kết quả đọc mà không có hậu tố _id hay x-ref (garden.zone, giống tv.tune/robot_vacuum.rooms), tham số ID trỏ sang
  app thật khác (contacts.contacts, giống ewallet.transfer), hai nguồn đọc (salon: dịch vụ + thợ), số dư phải kiểm trước
  (parking), dời lịch có start + end.
- Mô tả tham số dùng đúng câu chữ của catalog train ("ID ..., lấy từ app.tool", "ISO-8601 đầy đủ giây...", "Số nguyên,
  không dấu chấm hay phẩy, ví dụ 2", x-match fuzzy → "văn bản tự do" trong prompt).
- Tool của app tổng hợp là method của một lớp con `zaic_tools.App` gắn thẳng vào Session thật của mẫu chủ, nên kiểm schema,
  thực thi, deepcopy kết quả và state_hash đều là code của zaic_tools.

Mọi thứ dựng lại được từ metadata JSON lưu kèm mẫu: {"domain", "names", "state"} → `install(session, meta)`.
Chỉ đọc split train. Không đọc public_test.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

DT = "ISO-8601 đầy đủ giây và offset múi giờ, ví dụ 2026-11-11T05:30:00+07:00"
DATE = "YYYY-MM-DD, ví dụ 2026-11-29"
INT = "Số nguyên, không dấu chấm hay phẩy, ví dụ 2"
MONEY = "Số nguyên, không dấu chấm, phẩy hay đơn vị, ví dụ 1200000"
HELDOUT = {
    "appliances",
    "door_lock",
    "fitness",
    "learning",
    "library",
    "loyalty",
    "medication",
    "package_tracking",
    "subscriptions",
    "video_meeting",
}


@dataclass
class P:
    """Tham số của tool. kind: text|enum|int|money|datetime|date|bool|ref|contact|nameref."""

    canon: str
    kind: str
    names: tuple = ()  # biến thể tên tham số; trống = canon
    req: bool = False
    values: tuple = ()  # enum
    say: dict = field(default_factory=dict)  # enum value -> cách nói
    desc: str = ""
    ref: str = ""  # ref: collection đích; nameref: khoá danh sách trong status
    lo: int = 1
    hi: int = 10
    unit: str = ""
    fmt: bool = False  # thêm "format" date / date-time


@dataclass
class Coll:
    names: tuple  # ((coll, sing), ...)
    prefix: tuple
    noun: str
    key: str  # trường mang tên người dùng gọi
    items: list
    n: tuple = (3, 5)


@dataclass
class Tool:
    key: str
    names: tuple
    op: str  # list|search|status|create|update|act|set|topup
    desc: str
    coll: str = ""
    params: list = field(default_factory=list)
    say: list = field(default_factory=list)  # mẫu câu (write) hoặc câu hỏi (read)
    sets: dict = field(default_factory=dict)  # act: trường đặt lại; set: trường status
    qname: tuple = ("query",)
    qdesc: str = ""


@dataclass
class Domain:
    key: str
    ids: tuple
    vi: str
    group: str
    desc: tuple
    colls: dict
    tools: list
    status: dict | None = None  # trạng thái thiết bị / tài khoản (dict), nếu có
    status_key: str = "status"


def E(**kw):  # enum say helper
    return kw


DOMAINS: list[Domain] = [
    Domain(
        "pet",
        ("pet_care", "pet_clinic"),
        "Chăm sóc thú cưng",
        "health",
        ("Hồ sơ thú cưng, đặt lịch khám, tiêm phòng, spa", "Lịch khám và tiêm phòng cho chó mèo"),
        {
            "pets": Coll(
                (("pets", "pet"),),
                ("pet",),
                "thú cưng",
                "name",
                [
                    {"name": n, "species": s}
                    for n, s in [
                        ("Mích", "mèo"),
                        ("Bông", "chó"),
                        ("Lu", "chó"),
                        ("Mực", "chó"),
                        ("Miu", "mèo"),
                        ("Su", "mèo"),
                        ("Vàng", "chó"),
                        ("Na", "mèo"),
                        ("Kem", "chó"),
                    ]
                ],
                (2, 4),
            ),
            "visits": Coll(
                (("visits", "visit"), ("appointments", "appointment")),
                ("apt", "vis"),
                "lịch khám",
                "pet_name",
                [],
                (1, 3),
            ),
        },
        [
            Tool(
                "list_pets",
                ("list_pets", "get_pets"),
                "list",
                "Danh sách thú cưng",
                "pets",
                say=["nhà {p} đang khai mấy bé thú cưng trong app vậy", "xem giúp {p} danh sách thú cưng"],
            ),
            Tool(
                "list_visits",
                ("list_visits", "my_appointments"),
                "list",
                "Lịch khám đã đặt",
                "visits",
                say=["{p} có lịch khám thú cưng nào sắp tới không", "xem lịch khám của mấy bé giúp {p}"],
            ),
            Tool(
                "book",
                ("book_visit", "book_appointment"),
                "create",
                "Đặt lịch khám/tiêm/spa",
                "visits",
                [
                    P("pet_id", "ref", req=True, ref="pets"),
                    P(
                        "service",
                        "enum",
                        req=True,
                        values=("checkup", "vaccine", "grooming"),
                        say=E(
                            checkup=["khám tổng quát", "khám sức khoẻ"],
                            vaccine=["tiêm phòng", "chích ngừa"],
                            grooming=["tắm cắt tỉa lông", "đi spa"],
                        ),
                        desc="checkup: khám tổng quát; vaccine: tiêm phòng; grooming: tắm, cắt tỉa lông",
                    ),
                    P("start", "datetime", req=True, fmt=True),
                    P("note", "text", names=("note", "symptom")),
                ],
                say=[
                    "đặt lịch {service} cho bé {obj} {start}[, ghi chú {note}]",
                    "{start} cho bé {obj} đi {service} nha[, nhắn bác sĩ là {note}]",
                    "book giúp {p} lịch {service} cho {obj} vào {start}",
                ],
            ),
            Tool(
                "resched",
                ("reschedule_visit", "move_appointment"),
                "update",
                "Dời lịch khám",
                "visits",
                [
                    P("visit_id", "ref", req=True, ref="visits"),
                    P("start", "datetime", req=True),
                    P("end", "datetime", req=True),
                ],
            ),
            Tool(
                "cancel",
                ("cancel_visit", "cancel_appointment"),
                "act",
                "Huỷ lịch khám",
                "visits",
                [P("visit_id", "ref", req=True, ref="visits")],
                say=["huỷ lịch khám của bé {obj}", "bỏ lịch hẹn thú y của {obj} đi"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "car",
        ("car_care", "garage_booking"),
        "Bảo dưỡng xe",
        "mobility",
        ("Đặt lịch bảo dưỡng, rửa xe, thay nhớt", "Gara: lịch bảo dưỡng và sửa xe"),
        {
            "vehicles": Coll(
                (("vehicles", "vehicle"), ("cars", "car")),
                ("veh", "car"),
                "xe",
                "name",
                [
                    {"name": n, "plate": p}
                    for n, p in [
                        ("xe Vios", "30H-512.37"),
                        ("xe SH", "29D1-883.20"),
                        ("xe Wave", "59X3-120.45"),
                        ("xe CX5", "51K-276.81"),
                        ("xe Vision", "29E1-450.12"),
                        ("xe VF8", "30L-019.66"),
                        ("xe Lead", "59G2-731.08"),
                    ]
                ],
                (2, 3),
            ),
            "bookings": Coll((("bookings", "booking"),), ("bk", "svc"), "lịch bảo dưỡng", "vehicle_name", [], (1, 2)),
        },
        [
            Tool(
                "list_vehicles",
                ("list_vehicles", "my_cars"),
                "list",
                "Danh sách xe đã đăng ký",
                "vehicles",
                say=["{p} đăng ký mấy xe trong app gara rồi", "xem danh sách xe của {p}"],
            ),
            Tool(
                "list_bookings",
                ("list_bookings", "service_history"),
                "list",
                "Lịch bảo dưỡng đã đặt",
                "bookings",
                say=["{p} có hẹn gara nào chưa", "xem lịch bảo dưỡng xe giúp {p}"],
            ),
            Tool(
                "book",
                ("book_service", "schedule_service"),
                "create",
                "Đặt lịch bảo dưỡng/sửa xe",
                "bookings",
                [
                    P("vehicle_id", "ref", req=True, ref="vehicles"),
                    P(
                        "service",
                        "enum",
                        req=True,
                        values=("oil_change", "maintenance", "wash", "tire"),
                        say=E(
                            oil_change=["thay nhớt", "thay dầu"],
                            maintenance=["bảo dưỡng định kỳ", "bảo dưỡng"],
                            wash=["rửa xe"],
                            tire=["thay lốp", "vá lốp"],
                        ),
                    ),
                    P("time", "datetime", names=("time", "start"), req=True),
                    P("note", "text"),
                ],
                say=[
                    "đặt lịch {service} cho {obj} {time}[, dặn thợ {note}]",
                    "{time} mang {obj} đi {service} nha, đặt lịch giúp {p}",
                    "đặt gara {service} {obj} vào {time}",
                ],
            ),
            Tool(
                "cancel",
                ("cancel_booking", "cancel_service"),
                "act",
                "Huỷ lịch bảo dưỡng",
                "bookings",
                [P("booking_id", "ref", req=True, ref="bookings")],
                say=["huỷ lịch gara của {obj}", "bỏ cái hẹn bảo dưỡng {obj} đi"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "garden",
        ("garden", "smart_garden"),
        "Tưới cây thông minh",
        "home",
        ("Tưới cây tự động theo khu, bồn nước", "Hệ thống tưới: khu tưới, bồn nước, lịch tự động"),
        {},
        [
            Tool(
                "status",
                ("get_status", "status"),
                "status",
                "Bồn nước, chế độ tự động và danh sách khu tưới (zones)",
                say=[
                    "bồn nước tưới cây còn bao nhiêu phần trăm",
                    "hệ thống tưới đang để tự động hay tay vậy",
                    "xem tình trạng tưới cây giúp {p}",
                ],
            ),
            Tool(
                "water",
                ("water_zone", "start_watering"),
                "set",
                "Tưới ngay một khu",
                "",
                [
                    P(
                        "zone",
                        "nameref",
                        req=True,
                        ref="zones",
                        names=("zone", "area"),
                        desc='Tên khu tưới, phải có trong zones của {app}.{status}, ví dụ "ban công"',
                    ),
                    P("minutes", "int", lo=5, hi=30, unit="phút", desc="Số phút tưới. " + INT),
                ],
                say=[
                    "tưới {obj}[ {minutes}] giúp {p}",
                    "mở nước tưới {obj}[ trong {minutes}] nha",
                    "cho tưới {obj}[ chừng {minutes}] đi",
                ],
            ),
            Tool(
                "auto",
                ("set_auto", "auto_mode"),
                "set",
                "Bật/tắt tưới tự động",
                "",
                [P("enabled", "bool", req=True, desc="true để bật, false để tắt")],
                say=["{enabled} chế độ tưới tự động", "{enabled} tưới tự động giúp {p}"],
            ),
            Tool(
                "stop",
                ("stop_watering", "stop"),
                "set",
                "Dừng mọi vòi đang tưới",
                "",
                [],
                say=["tắt hết vòi tưới đi", "dừng tưới cây lại giúp {p}"],
            ),
        ],
        status={
            "zones": ["vườn trước", "ban công", "sân sau", "vườn rau", "giàn bầu", "chậu lan", "bồn hoa cổng"],
            "tank_percent": 0,
            "auto": False,
            "watering": None,
        },
    ),
    Domain(
        "water",
        ("drinking_water", "water_delivery"),
        "Giao nước uống",
        "shopping",
        ("Đặt nước bình, nước thùng giao tận nhà", "Nước uống đóng bình giao tận nơi"),
        {
            "products": Coll(
                (("products", "product"),),
                ("prod", "sp"),
                "sản phẩm",
                "name",
                [
                    {"name": n, "price": p}
                    for n, p in [
                        ("bình Lavie 19 lít", 65000),
                        ("bình Aquafina 19 lít", 70000),
                        ("thùng Vĩnh Hảo 24 chai", 98000),
                        ("bình ion kiềm 20 lít", 120000),
                        ("bình Lavie 5 lít", 32000),
                        ("thùng Dasani 24 chai", 105000),
                    ]
                ],
                (3, 5),
            ),
            "orders": Coll((("orders", "order"),), ("ord", "dh"), "đơn nước", "product_name", [], (1, 2)),
        },
        [
            Tool(
                "list_products",
                ("list_products", "catalog"),
                "list",
                "Sản phẩm và giá",
                "products",
                say=["bình Lavie bao nhiêu tiền vậy", "xem bên nước uống có những loại gì"],
            ),
            Tool(
                "orders",
                ("order_history", "my_orders"),
                "list",
                "Đơn đã đặt",
                "orders",
                say=["đơn nước của {p} giao chưa", "xem mấy đơn nước gần đây"],
            ),
            Tool(
                "order",
                ("place_order", "order"),
                "create",
                "Đặt nước",
                "orders",
                [
                    P("product_id", "ref", req=True, ref="products"),
                    P("quantity", "int", req=True, lo=1, hi=6, unit="", desc="Số lượng. " + INT),
                    P(
                        "deliver_at",
                        "datetime",
                        names=("deliver_at", "delivery_time"),
                        desc="Giờ giao mong muốn, " + DT + "; bỏ trống để giao sớm nhất",
                    ),
                ],
                say=[
                    "đặt {quantity} {obj}[ giao {deliver_at}]",
                    "gọi thêm {quantity} {obj} nha[, giao {deliver_at}]",
                    "cho {p} {quantity} {obj}[ {deliver_at}]",
                ],
            ),
            Tool(
                "cancel",
                ("cancel_order", "cancel"),
                "act",
                "Huỷ đơn chưa giao",
                "orders",
                [P("order_id", "ref", req=True, ref="orders")],
                say=["huỷ đơn {obj} đi", "thôi huỷ cái đơn {obj} giúp {p}"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "clean",
        ("home_cleaning", "maid_service"),
        "Giúp việc theo giờ",
        "home",
        ("Đặt người giúp việc dọn nhà theo giờ", "Dọn dẹp nhà theo giờ"),
        {
            "helpers": Coll(
                (("helpers", "helper"), ("cleaners", "cleaner")),
                ("hp", "cl"),
                "người giúp việc",
                "name",
                [
                    {"name": n, "rating": r}
                    for n, r in [
                        ("chị Hạnh", 4.8),
                        ("cô Tư", 4.9),
                        ("chị Mai Anh", 4.6),
                        ("cô Liên", 4.7),
                        ("chị Thảo", 4.5),
                        ("cô Sáu", 4.8),
                        ("chị Ngọc Hà", 4.7),
                    ]
                ],
                (3, 5),
            ),
            "jobs": Coll(
                (("jobs", "job"), ("bookings", "booking")), ("job", "bk"), "lịch dọn", "helper_name", [], (1, 2)
            ),
        },
        [
            Tool(
                "find",
                ("search_helpers", "find_cleaner"),
                "search",
                "Tìm người giúp việc theo tên; không truyền query thì trả tất cả",
                "helpers",
                qname=("query", "name"),
                qdesc="Tên người giúp việc",
            ),
            Tool(
                "jobs",
                ("list_jobs", "my_bookings"),
                "list",
                "Lịch dọn đã đặt",
                "jobs",
                say=["tuần này {p} có đặt dọn nhà buổi nào chưa", "xem lịch giúp việc đã đặt"],
            ),
            Tool(
                "book",
                ("book_cleaning", "book"),
                "create",
                "Đặt dọn nhà",
                "jobs",
                [
                    P("helper_id", "ref", req=True, ref="helpers"),
                    P("start", "datetime", req=True, fmt=True),
                    P("hours", "int", req=True, lo=2, hi=4, unit="tiếng", desc="Số giờ dọn. " + INT),
                ],
                say=[
                    "đặt {obj} dọn nhà {start}[ {hours}]",
                    "{start} gọi {obj} qua dọn nhà[ {hours}] nha",
                    "book {obj} lau dọn[ {hours}] vào {start}",
                ],
            ),
            Tool(
                "resched",
                ("reschedule_job", "move_booking"),
                "update",
                "Dời lịch dọn",
                "jobs",
                [
                    P("job_id", "ref", req=True, ref="jobs"),
                    P("start", "datetime", req=True),
                    P("end", "datetime", req=True),
                ],
            ),
            Tool(
                "cancel",
                ("cancel_job", "cancel_booking"),
                "act",
                "Huỷ lịch dọn",
                "jobs",
                [P("job_id", "ref", req=True, ref="jobs")],
                say=["huỷ buổi dọn nhà của {obj}", "báo {obj} khỏi qua dọn nữa, huỷ lịch giúp {p}"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "school",
        ("school_link", "school_contact"),
        "Sổ liên lạc điện tử",
        "personal",
        ("Sổ liên lạc: xin nghỉ học, khoản thu của trường", "Liên lạc với trường của con"),
        {
            "children": Coll(
                (("children", "child"), ("students", "student")),
                ("kid", "hs"),
                "con",
                "name",
                [
                    {"name": n, "class": c}
                    for n, c in [
                        ("bé Na", "lớp 2A"),
                        ("bé Bin", "lớp lá 1"),
                        ("bé Su", "lớp 4C"),
                        ("bé Tôm", "lớp mầm 2"),
                        ("bé Cà Rốt", "lớp chồi 3"),
                        ("bé Ốc", "lớp 1B"),
                    ]
                ],
                (1, 3),
            ),
            "fees": Coll(
                (("fees", "fee"), ("payments", "payment")),
                ("fee", "kt"),
                "khoản thu",
                "title",
                [
                    {"title": t, "amount": a}
                    for t, a in [
                        ("học phí tháng mười một", 1800000),
                        ("tiền ăn bán trú", 750000),
                        ("tiền dã ngoại", 350000),
                        ("quỹ lớp", 200000),
                        ("tiền đồng phục", 420000),
                        ("tiền học bơi", 600000),
                    ]
                ],
                (2, 4),
            ),
        },
        [
            Tool("children", ("list_children", "my_children"), "list", "Danh sách con và lớp", "children"),
            Tool(
                "leave",
                ("send_leave_note", "request_leave"),
                "create",
                "Gửi đơn xin nghỉ học cho con",
                "leaves",
                [
                    P("child_id", "ref", req=True, ref="children"),
                    P("date", "date", req=True, fmt=True),
                    P("reason", "text", req=True),
                ],
                say=[
                    "xin cho {obj} nghỉ học {date}, lý do {reason}",
                    "{date} {obj} nghỉ học nha, gửi cô là {reason}",
                    "gửi đơn xin phép cho {obj} nghỉ {date} vì {reason}",
                ],
            ),
            Tool(
                "fees",
                ("list_fees", "pending_fees"),
                "list",
                "Khoản thu của trường (chưa đóng và đã đóng)",
                "fees",
                [P("status", "enum", values=("unpaid", "paid"), desc="unpaid: chưa đóng; paid: đã đóng")],
                say=["trường của con còn khoản nào chưa đóng không", "tháng này phải đóng cho trường bao nhiêu"],
            ),
            Tool(
                "pay",
                ("pay_fee", "pay"),
                "act",
                "Đóng một khoản thu",
                "fees",
                [
                    P("fee_id", "ref", req=True, ref="fees"),
                    P(
                        "source",
                        "enum",
                        req=True,
                        values=("bank", "ewallet"),
                        say=E(
                            bank=["bằng tài khoản ngân hàng", "qua ngân hàng"], ewallet=["bằng ví", "bằng ví điện tử"]
                        ),
                    ),
                ],
                say=["đóng {obj} {source}", "trả {obj} cho trường {source} giúp {p}"],
                sets={"status": "paid"},
            ),
        ],
    ),
    Domain(
        "salon",
        ("hair_salon", "salon_booking"),
        "Đặt lịch salon tóc",
        "personal",
        ("Dịch vụ tóc, thợ, đặt lịch salon", "Salon: dịch vụ, thợ và lịch hẹn"),
        {
            "services": Coll(
                (("services", "service"),),
                ("svc", "dv"),
                "dịch vụ",
                "name",
                [
                    {"name": n, "price": p, "minutes": m}
                    for n, p, m in [
                        ("cắt tóc nam", 120000, 30),
                        ("gội đầu dưỡng sinh", 150000, 60),
                        ("nhuộm tóc", 650000, 120),
                        ("uốn tóc", 800000, 150),
                        ("làm móng", 200000, 60),
                        ("hấp dầu", 250000, 60),
                    ]
                ],
                (3, 5),
            ),
            "stylists": Coll(
                (("stylists", "stylist"),),
                ("sty", "tho"),
                "thợ",
                "name",
                [{"name": n} for n in ["anh Tuấn", "chị Vy", "anh Khang", "chị Ngân", "anh Lộc", "chị Diễm"]],
                (2, 4),
            ),
            "appointments": Coll(
                (("appointments", "appointment"),), ("ap", "lh"), "lịch hẹn", "service_name", [], (0, 1)
            ),
        },
        [
            Tool(
                "services",
                ("list_services", "services"),
                "list",
                "Dịch vụ, giá và thời lượng (phút)",
                "services",
                say=["gội đầu dưỡng sinh bên salon giá bao nhiêu", "salon có những dịch vụ gì vậy"],
            ),
            Tool("stylists", ("list_stylists", "stylists"), "list", "Thợ đang làm", "stylists"),
            Tool(
                "book",
                ("book", "book_appointment"),
                "create",
                "Đặt lịch",
                "appointments",
                [
                    P("service_id", "ref", req=True, ref="services"),
                    P("start", "datetime", req=True, fmt=True),
                    P("stylist_id", "ref", ref="stylists", desc="bỏ trống để salon tự xếp thợ"),
                ],
                say=[
                    "đặt lịch {obj} {start}[ với {stylist_id}]",
                    "{start} {p} qua {obj} nha[, chọn {stylist_id}]",
                    "book {obj}[ thợ {stylist_id}] lúc {start}",
                ],
            ),
            Tool("mine", ("my_appointments", "list_appointments"), "list", "Lịch hẹn đã đặt", "appointments"),
            Tool(
                "cancel",
                ("cancel", "cancel_appointment"),
                "act",
                "Huỷ lịch hẹn",
                "appointments",
                [P("appointment_id", "ref", req=True, ref="appointments")],
                say=["huỷ lịch {obj} ở salon", "bỏ hẹn {obj} giúp {p}"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "gift",
        ("gift_shop", "quick_gift"),
        "Gửi quà tận nơi",
        "shopping",
        ("Gửi hoa, bánh, quà tặng tới người thân", "Quà tặng giao tận nơi"),
        {
            "gifts": Coll(
                (("gifts", "gift"), ("products", "product")),
                ("gift", "qt"),
                "món quà",
                "name",
                [
                    {"name": n, "price": p}
                    for n, p in [
                        ("bó hoa hồng", 350000),
                        ("giỏ trái cây", 450000),
                        ("bánh kem socola", 280000),
                        ("hộp bánh quy bơ", 190000),
                        ("chậu lan hồ điệp", 900000),
                        ("hộp trà sen", 320000),
                    ]
                ],
                (3, 5),
            ),
            "orders": Coll((("orders", "order"),), ("go", "gd"), "đơn quà", "gift_name", [], (0, 1)),
        },
        [
            Tool(
                "find",
                ("search_gifts", "find_gift"),
                "search",
                "Tìm quà theo tên; không truyền query thì trả tất cả",
                "gifts",
                qname=("query", "keyword"),
                qdesc="Tên món quà, ví dụ hoa hồng",
            ),
            Tool(
                "send",
                ("send_gift", "order_gift"),
                "create",
                "Gửi quà cho một liên hệ",
                "orders",
                [
                    P("gift_id", "ref", req=True, ref="gifts"),
                    P("contact_id", "contact", req=True, names=("contact_id", "recipient_id")),
                    P("message", "text", names=("message", "card_text")),
                    P("deliver_date", "date", desc="Ngày giao, " + DATE + "; bỏ trống để giao hôm nay"),
                ],
                say=[
                    "gửi {obj} cho {contact_id}[ {deliver_date}][, thiệp ghi {message}]",
                    "đặt {obj} tặng {contact_id}[, giao {deliver_date}][, kèm lời chúc {message}]",
                ],
            ),
            Tool("orders", ("list_orders", "my_orders"), "list", "Đơn quà đã gửi", "orders"),
        ],
    ),
    Domain(
        "split",
        ("split_bill", "group_pay"),
        "Chia tiền nhóm",
        "finance",
        ("Đòi tiền, chia hoá đơn với bạn bè", "Ghi nợ và nhắc trả tiền"),
        {"requests": Coll((("requests", "request"),), ("req", "yc"), "yêu cầu trả tiền", "contact_name", [], (1, 3))},
        [
            Tool(
                "list",
                ("list_requests", "pending_requests"),
                "list",
                "Các yêu cầu trả tiền đã gửi",
                "requests",
                say=["ai còn nợ tiền {p} chưa trả vậy", "xem mấy khoản {p} đòi tiền giúp"],
            ),
            Tool(
                "request",
                ("request_money", "create_request"),
                "create",
                "Gửi yêu cầu trả tiền tới một liên hệ",
                "requests",
                [
                    P("contact_id", "contact", req=True),
                    P("amount", "money", req=True),
                    P("note", "text", names=("note", "reason")),
                ],
                say=[
                    "đòi {contact_id} {amount}[ tiền {note}]",
                    "gửi yêu cầu {contact_id} trả {p} {amount}[, ghi là {note}]",
                    "nhắn đòi {amount}[ {note}] từ {contact_id}",
                ],
            ),
            Tool(
                "remind",
                ("remind_request", "nudge"),
                "act",
                "Nhắc lại một yêu cầu chưa trả",
                "requests",
                [P("request_id", "ref", req=True, ref="requests")],
                say=["nhắc {obj} trả tiền {p} đi", "giục {obj} chuyển tiền giúp {p}"],
                sets={"reminded": True},
            ),
            Tool(
                "cancel",
                ("cancel_request", "close_request"),
                "act",
                "Huỷ yêu cầu trả tiền",
                "requests",
                [P("request_id", "ref", req=True, ref="requests")],
                say=["thôi huỷ khoản đòi tiền {obj}", "xoá yêu cầu trả tiền của {obj}"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "parking",
        ("parking_pass", "smart_parking"),
        "Gửi xe tháng",
        "mobility",
        ("Tài khoản gửi xe, vé tháng theo xe", "Vé tháng bãi xe chung cư"),
        {
            "vehicles": Coll(
                (("vehicles", "vehicle"),),
                ("pv", "xe"),
                "xe",
                "name",
                [
                    {"name": n}
                    for n in [
                        "xe máy Vision",
                        "ô tô Mazda",
                        "xe Lead",
                        "ô tô Kia Morning",
                        "xe đạp điện",
                        "xe Air Blade",
                    ]
                ],
                (2, 3),
            )
        },
        [
            Tool(
                "account",
                ("get_account", "balance"),
                "status",
                "Số dư tài khoản gửi xe và giá vé tháng (price_per_month)",
                say=["tài khoản gửi xe còn bao nhiêu tiền", "vé tháng gửi xe giá bao nhiêu"],
            ),
            Tool("vehicles", ("list_vehicles", "vehicles"), "list", "Xe đã đăng ký", "vehicles"),
            Tool(
                "buy",
                ("buy_pass", "renew_pass"),
                "set",
                "Mua/gia hạn vé tháng, trừ vào số dư",
                "vehicles",
                [
                    P("vehicle_id", "ref", req=True, ref="vehicles"),
                    P("months", "int", req=True, lo=1, hi=6, unit="tháng", desc="Số tháng. " + INT),
                ],
                say=["gia hạn vé tháng cho {obj} {months}", "mua vé gửi {obj} {months} nha"],
            ),
            Tool(
                "topup",
                ("top_up", "deposit"),
                "topup",
                "Nạp tiền vào tài khoản gửi xe",
                "",
                [
                    P("amount", "money", req=True),
                    P(
                        "source",
                        "enum",
                        req=True,
                        values=("bank", "ewallet"),
                        say=E(bank=["từ ngân hàng", "bằng tài khoản ngân hàng"], ewallet=["từ ví", "bằng ví điện tử"]),
                    ),
                ],
                say=["nạp {amount} vào tài khoản gửi xe {source}", "nạp thêm {amount} tiền gửi xe {source} nha"],
            ),
        ],
        status={"balance": 0, "price_per_month": 0},
    ),
    Domain(
        "cinema",
        ("movie_tickets", "cinema_booking"),
        "Vé xem phim",
        "media",
        ("Suất chiếu và đặt vé xem phim", "Đặt vé rạp chiếu phim"),
        {
            "showtimes": Coll((("showtimes", "showtime"),), ("st", "sc"), "suất chiếu", "movie", [], (4, 6)),
            "tickets": Coll((("tickets", "ticket"),), ("tk", "ve"), "vé", "movie", [], (0, 1)),
        },
        [
            Tool(
                "find",
                ("search_showtimes", "find_showtimes"),
                "search",
                "Tìm suất chiếu theo tên phim; không truyền query thì trả tất cả",
                "showtimes",
                qname=("movie", "query"),
                qdesc="Tên phim",
            ),
            Tool(
                "book",
                ("book_tickets", "buy_tickets"),
                "create",
                "Đặt vé cho một suất chiếu",
                "tickets",
                [
                    P("showtime_id", "ref", req=True, ref="showtimes"),
                    P("seats", "int", req=True, lo=1, hi=5, unit="vé", desc="Số ghế. " + INT),
                    P(
                        "seat_type",
                        "enum",
                        values=("standard", "vip"),
                        say=E(standard=["ghế thường"], vip=["ghế VIP", "ghế vip"]),
                    ),
                ],
                say=[
                    "đặt {seats} {obj} suất {start}[ {seat_type}]",
                    "lấy {seats} {obj} lúc {start}[, {seat_type}] nha",
                ],
            ),
            Tool("mine", ("my_tickets", "tickets"), "list", "Vé đã đặt", "tickets"),
            Tool(
                "cancel",
                ("cancel_ticket", "refund"),
                "act",
                "Huỷ vé",
                "tickets",
                [P("ticket_id", "ref", req=True, ref="tickets")],
                say=["huỷ vé {obj}", "trả lại vé {obj} giúp {p}"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "clinic",
        ("clinic_booking", "doctor_booking"),
        "Đặt khám bệnh",
        "health",
        ("Tìm bác sĩ, đặt và dời lịch khám", "Phòng khám: bác sĩ và lịch hẹn"),
        {
            "doctors": Coll(
                (("doctors", "doctor"),),
                ("dr", "bs"),
                "bác sĩ",
                "name",
                [
                    {"name": n, "specialty": s}
                    for n, s in [
                        ("bác sĩ Lan", "nhi"),
                        ("bác sĩ Minh Quân", "da liễu"),
                        ("bác sĩ Hoàng", "răng hàm mặt"),
                        ("bác sĩ Thu Trang", "tai mũi họng"),
                        ("bác sĩ Khôi", "tim mạch"),
                        ("bác sĩ Diệp", "mắt"),
                    ]
                ],
                (3, 5),
            ),
            "appointments": Coll(
                (("appointments", "appointment"), ("visits", "visit")),
                ("appt", "lk"),
                "lịch khám",
                "doctor_name",
                [],
                (1, 2),
            ),
        },
        [
            Tool(
                "find",
                ("search_doctors", "find_doctor"),
                "search",
                "Tìm bác sĩ theo tên hoặc chuyên khoa; không truyền query thì trả tất cả",
                "doctors",
                qname=("query",),
                qdesc="Tên bác sĩ hoặc chuyên khoa",
            ),
            Tool(
                "book",
                ("book_appointment", "book"),
                "create",
                "Đặt lịch khám",
                "appointments",
                [
                    P("doctor_id", "ref", req=True, ref="doctors"),
                    P("start", "datetime", req=True),
                    P("symptom", "text", names=("symptom", "reason")),
                ],
                say=[
                    "đặt khám {obj} {start}[, triệu chứng {symptom}]",
                    "{start} {p} muốn khám {obj}[ vì {symptom}]",
                    "lấy lịch {obj} vào {start}",
                ],
            ),
            Tool(
                "mine",
                ("my_appointments", "appointments"),
                "list",
                "Lịch khám đã đặt",
                "appointments",
                say=["{p} có lịch khám bác sĩ nào sắp tới không"],
            ),
            Tool(
                "resched",
                ("reschedule", "move_appointment"),
                "update",
                "Dời lịch khám",
                "appointments",
                [
                    P("appointment_id", "ref", req=True, ref="appointments"),
                    P("start", "datetime", req=True),
                    P("end", "datetime", req=True),
                ],
            ),
            Tool(
                "cancel",
                ("cancel_appointment", "cancel"),
                "act",
                "Huỷ lịch khám",
                "appointments",
                [P("appointment_id", "ref", req=True, ref="appointments")],
                say=["huỷ lịch khám với {obj}", "bỏ hẹn khám {obj} đi"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
    Domain(
        "heater",
        ("water_heater", "smart_heater"),
        "Bình nóng lạnh",
        "home",
        ("Bật/tắt, đặt nhiệt độ bình nóng lạnh", "Điều khiển bình nước nóng"),
        {},
        [
            Tool(
                "status",
                ("get_status", "status"),
                "status",
                "Trạng thái bật/tắt, nhiệt độ hiện tại và nhiệt độ đặt",
                say=["bình nóng lạnh đang bật hay tắt", "nước trong bình nóng chưa, bao nhiêu độ rồi"],
            ),
            Tool(
                "power",
                ("set_power", "power"),
                "set",
                "Bật/tắt bình",
                "",
                [P("power", "enum", req=True, values=("on", "off"), say=E(on=["bật"], off=["tắt"]))],
                say=["{power} bình nóng lạnh", "{power} bình nước nóng giúp {p}"],
            ),
            Tool(
                "temp",
                ("set_temperature", "set_temp"),
                "set",
                "Đặt nhiệt độ nước (tự bật bình)",
                "",
                [
                    P("temperature", "int", req=True, lo=35, hi=75, unit="độ", desc="Độ C, từ 35 đến 75. " + INT),
                    P(
                        "mode",
                        "enum",
                        values=("eco", "boost"),
                        say=E(eco=["chế độ tiết kiệm"], boost=["chế độ nhanh", "đun nhanh"]),
                        desc="eco: tiết kiệm điện; boost: đun nhanh",
                    ),
                ],
                say=["chỉnh bình nóng lạnh lên {temperature}[, {mode}]", "đặt nước nóng {temperature}[ {mode}] nha"],
            ),
        ],
        status={"power": "off", "current_temp": 30, "target_temp": 45, "mode": "eco"},
    ),
    Domain(
        "building",
        ("building_service", "apartment_service"),
        "Dịch vụ chung cư",
        "home",
        ("Báo hỏng, đặt tiện ích chung cư", "Ban quản lý toà nhà: phản ánh và tiện ích"),
        {
            "tickets": Coll(
                (("tickets", "ticket"), ("requests", "request")), ("tk", "pa"), "phản ánh", "description", [], (0, 2)
            ),
            "amenities": Coll((("bookings", "booking"),), ("am", "ti"), "lịch tiện ích", "amenity", [], (0, 1)),
        },
        [
            Tool(
                "list",
                ("list_tickets", "my_requests"),
                "list",
                "Phản ánh đã gửi và trạng thái",
                "tickets",
                say=["mấy cái phản ánh {p} gửi ban quản lý xử lý chưa", "xem phản ánh đã gửi giúp {p}"],
            ),
            Tool(
                "report",
                ("report_issue", "create_ticket"),
                "create",
                "Gửi phản ánh/báo hỏng cho ban quản lý",
                "tickets",
                [
                    P(
                        "category",
                        "enum",
                        req=True,
                        values=("electric", "water", "elevator", "cleaning", "noise"),
                        say=E(
                            electric=["hỏng điện", "mất điện"],
                            water=["rò nước", "mất nước"],
                            elevator=["thang máy hỏng"],
                            cleaning=["rác chưa dọn"],
                            noise=["ồn ào"],
                        ),
                        desc="electric: điện; water: nước; elevator: thang máy; cleaning: vệ sinh; noise: tiếng ồn",
                    ),
                    P("description", "text", req=True, names=("description", "detail")),
                    P("preferred_time", "datetime", desc="Giờ muốn kỹ thuật qua, " + DT),
                ],
                say=[
                    "báo ban quản lý {category}, {description}[, nhờ qua {preferred_time}]",
                    "gửi phản ánh {category} nha: {description}[, hẹn {preferred_time}]",
                ],
            ),
            Tool(
                "amenity",
                ("book_amenity", "reserve_facility"),
                "create",
                "Đặt tiện ích chung (BBQ, gym, hồ bơi, phòng họp)",
                "amenities",
                [
                    P(
                        "amenity",
                        "enum",
                        req=True,
                        values=("bbq", "pool", "gym", "meeting_room"),
                        say=E(
                            bbq=["khu nướng BBQ", "chỗ nướng BBQ"],
                            pool=["hồ bơi"],
                            gym=["phòng gym"],
                            meeting_room=["phòng sinh hoạt chung", "phòng họp chung"],
                        ),
                    ),
                    P("start", "datetime", req=True, fmt=True),
                    P("hours", "int", lo=1, hi=3, unit="tiếng", desc="Số giờ. " + INT),
                ],
                say=["đặt {amenity} {start}[ {hours}]", "giữ {amenity} cho {p} vào {start}[ trong {hours}]"],
            ),
            Tool(
                "cancel",
                ("cancel_ticket", "close_request"),
                "act",
                "Huỷ phản ánh",
                "tickets",
                [P("ticket_id", "ref", req=True, ref="tickets")],
                say=["huỷ phản ánh {obj} đi"],
                sets={"status": "cancelled"},
            ),
        ],
    ),
]
BY_KEY = {d.key: d for d in DOMAINS}


# ---------------------------------------------------------------------------------------------------------------------
def choose_names(dom: Domain, rng: random.Random, taken: set[str]) -> dict:
    """Mọi lựa chọn ngẫu nhiên về tên (app, collection, tool, tham số, tiền tố ID): đủ để dựng lại app."""
    apps = [a for a in dom.ids if a not in taken]
    if not apps:
        raise ValueError("hết tên app")
    names = {
        "app": rng.choice(apps),
        "colls": {},
        "tools": {},
        "params": {},
        "seq_ids": rng.random() < 0.8,
        "fmt": {},
        "shuffle": rng.random() < 0.3,
        "req_note": rng.random() < 0.3,
    }
    for ck, c in dom.colls.items():
        coll, sing = rng.choice(c.names)
        names["colls"][ck] = {"name": coll, "sing": sing, "prefix": rng.choice(c.prefix)}
    if "leaves" not in names["colls"] and dom.key == "school":
        names["colls"]["leaves"] = {"name": "leave_notes", "sing": "leave_note", "prefix": "ln"}
    for t in dom.tools:
        names["tools"][t.key] = rng.choice(t.names)
        if t.op == "search":
            names["params"][f"{t.key}.$q"] = rng.choice(t.qname)
        for p in t.params:
            names["params"][f"{t.key}.{p.canon}"] = rng.choice(p.names) if p.names else p.canon
            if p.kind in ("datetime", "date"):
                names["fmt"][f"{t.key}.{p.canon}"] = p.fmt or rng.random() < 0.4
    return names


def coll_meta(dom: Domain, names: dict, ck: str) -> dict:
    return names["colls"][ck]


def finder_of(dom: Domain, ck: str) -> Tool | None:
    return next((t for t in dom.tools if t.coll == ck and t.op in ("list", "search")), None)


def pname(names: dict, t: Tool, p: P) -> str:
    return names["params"][f"{t.key}.{p.canon}"]


def catalog(dom: Domain, names: dict) -> dict:
    app = names["app"]
    st_tool = next((t for t in dom.tools if t.op == "status"), None)
    tools = []
    for t in dom.tools:
        props, req = {}, []
        if t.op == "search":
            props[names["params"][f"{t.key}.$q"]] = {"type": "string", "description": t.qdesc}
        for p in t.params:
            pn = pname(names, t, p)
            s: dict
            if p.kind == "text":
                s = {"type": "string", "x-match": "fuzzy"}
            elif p.kind == "enum":
                s = {"type": "string", "enum": list(p.values)}
            elif p.kind == "int":
                s = {"type": "integer", "description": p.desc or INT}
                if p.canon == "temperature":
                    s.update(minimum=p.lo, maximum=p.hi)
                else:
                    s["exclusiveMinimum"] = 0
            elif p.kind == "money":
                s = {"type": "integer", "description": MONEY, "exclusiveMinimum": 0}
            elif p.kind == "bool":
                s = {"type": "boolean"}
            elif p.kind in ("date", "datetime"):
                s = {"type": "string", "description": DT if p.kind == "datetime" else DATE}
                if names["fmt"].get(f"{t.key}.{p.canon}"):
                    s["format"] = "date-time" if p.kind == "datetime" else "date"
            elif p.kind == "ref":
                cm = names["colls"][p.ref]
                f = finder_of(dom, p.ref)
                s = {
                    "type": "string",
                    "description": f"ID {dom.colls[p.ref].noun}, lấy từ {app}.{names['tools'][f.key]}",
                    "x-ref": f"{app}.{cm['name']}",
                }
            elif p.kind == "contact":
                s = {
                    "type": "string",
                    "description": "ID liên hệ, lấy từ contacts.search",
                    "x-ref": "contacts.contacts",
                }
            elif p.kind == "nameref":
                s = {"type": "string"}
            else:
                raise ValueError(p.kind)
            if p.desc and p.kind != "int":
                d = p.desc.format(app=app, status=names["tools"][st_tool.key] if st_tool else "")
                if p.kind in ("ref", "text"):
                    s["description"] = (s["description"] + "; " + d) if s.get("description") else d
                else:  # enum, nameref, bool, date(time) có mô tả riêng đầy đủ
                    s["description"] = d
            props[pn] = s
            if p.req:
                req.append(pn)
        desc = t.desc + (
            f" (bắt buộc: {', '.join(req)})"
            if req and names["req_note"] and t.op not in ("list", "search", "status")
            else ""
        )
        ret = ""
        if t.op in ("list", "search"):
            ret = f"{names['colls'][t.coll]['name']}[]"
        elif t.op == "status":
            ret = ", ".join(k for k in (dom.status or {}) if k != "watering")
        tools.append(
            {
                "name": f"{app}.{names['tools'][t.key]}",
                "kind": "read" if t.op in ("list", "search", "status") else "write",
                "description": desc,
                "parameters": {"type": "object", "properties": props, "required": req},
                "returns": ret,
                "app": app,
            }
        )
    if names["shuffle"]:
        random.Random(app + str(len(tools))).shuffle(tools)
    return {"id": app, "name_vi": dom.vi, "group": dom.group, "description": dom.desc[0], "tools": tools}


# ---------------------------------------------------------------------------------------------------------------------
def _iso(d: datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:00+07:00")


def _ids(prefix: str, n: int, seq: bool, rng: random.Random) -> list[str]:
    if seq:
        return [f"{prefix}_{i + 1:02d}" for i in range(n)]
    return [f"{prefix}_{x}" for x in rng.sample(range(101, 999), n)]


def future_slot(now: datetime, rng: random.Random, lo=0, hi=9) -> datetime:
    for _ in range(30):
        k = rng.randint(lo, hi)
        h = rng.randint(7, 20)
        m = rng.choice([0, 0, 0, 30, 30, 15])
        d = datetime(now.year, now.month, now.day, h, m) + timedelta(days=k)
        if d > now.replace(tzinfo=None) + timedelta(hours=2):
            return d
    return datetime(now.year, now.month, now.day, 9, 0) + timedelta(days=2)


def initial_state(dom: Domain, names: dict, now: datetime, rng: random.Random, contacts: list[dict]) -> dict:
    """State ban đầu: các collection có mục thật, collection con trỏ tới mục cha có thật."""
    st: dict = {}
    seq = names["seq_ids"]
    for ck, c in dom.colls.items():
        if c.items:
            n = rng.randint(*c.n)
            items = rng.sample(c.items, min(n, len(c.items)))
            cm = names["colls"][ck]
            st[cm["name"]] = [
                {"id": i, **copy.deepcopy(x)} for i, x in zip(_ids(cm["prefix"], len(items), seq, rng), items)
            ]
    nowd = now.replace(tzinfo=None)

    def coll(ck):
        return st.setdefault(names["colls"][ck]["name"], [])

    def mk(ck, n, make):
        cm = names["colls"][ck]
        out = []
        for i in _ids(cm["prefix"], n, seq, rng):
            x = make()
            if x:
                out.append({"id": i, **x})
        st[cm["name"]] = out

    k = dom.key
    if k == "pet":
        pets = coll("pets")

        def visit():
            p = rng.choice(pets)
            s = future_slot(nowd, rng)
            dur = rng.choice([30, 60])
            return {
                "pet_id": p["id"],
                "pet_name": p["name"],
                "service": rng.choice(["checkup", "vaccine", "grooming"]),
                "start": _iso(s),
                "end": _iso(s + timedelta(minutes=dur)),
                "status": "booked",
            }

        mk("visits", rng.randint(*dom.colls["visits"].n), visit)
    elif k == "car":
        vs = coll("vehicles")

        def bk():
            v = rng.choice(vs)
            s = future_slot(nowd, rng)
            return {
                "vehicle_id": v["id"],
                "vehicle_name": v["name"],
                "service": rng.choice(["oil_change", "maintenance", "wash", "tire"]),
                "time": _iso(s),
                "status": "booked",
            }

        mk("bookings", rng.randint(*dom.colls["bookings"].n), bk)
    elif k == "clean":
        hs = coll("helpers")

        def job():
            h = rng.choice(hs)
            s = future_slot(nowd, rng)
            hrs = rng.choice([2, 3, 4])
            return {
                "helper_id": h["id"],
                "helper_name": h["name"],
                "start": _iso(s),
                "end": _iso(s + timedelta(hours=hrs)),
                "hours": hrs,
                "status": "booked",
            }

        mk("jobs", rng.randint(*dom.colls["jobs"].n), job)
    elif k == "school":
        kids = coll("children")
        for f in coll("fees"):
            c = rng.choice(kids)
            f.update(child_id=c["id"], child_name=c["name"], status=rng.choice(["unpaid", "unpaid", "paid"]))
        st[names["colls"]["leaves"]["name"]] = []
    elif k == "water":
        ps = coll("products")

        def od():
            p = rng.choice(ps)
            return {
                "product_id": p["id"],
                "product_name": p["name"],
                "quantity": rng.randint(1, 3),
                "status": rng.choice(["pending", "delivered"]),
            }

        mk("orders", rng.randint(*dom.colls["orders"].n), od)
    elif k == "salon":
        sv, sy = coll("services"), coll("stylists")

        def ap():
            s = rng.choice(sv)
            t = future_slot(nowd, rng)
            return {
                "service_id": s["id"],
                "service_name": s["name"],
                "stylist_id": rng.choice(sy)["id"],
                "start": _iso(t),
                "end": _iso(t + timedelta(minutes=s["minutes"])),
                "status": "booked",
            }

        mk("appointments", rng.randint(*dom.colls["appointments"].n), ap)
    elif k == "gift":
        coll("orders")
    elif k == "split":
        pool = [c for c in contacts] or [{"id": "c_01", "name": "Minh"}]

        def rq():
            c = rng.choice(pool)
            return {
                "contact_id": c["id"],
                "contact_name": c["name"],
                "amount": rng.choice([50000, 120000, 200000, 350000, 500000]),
                "note": rng.choice(["tiền ăn trưa", "tiền vé xem phim", "tiền điện tháng trước", "tiền taxi"]),
                "status": "pending",
                "reminded": False,
            }

        mk("requests", rng.randint(*dom.colls["requests"].n), rq)
        # trùng tên liên hệ trong các yêu cầu làm câu nói mơ hồ: bỏ bớt
        seen, keep = set(), []
        for r in st[names["colls"]["requests"]["name"]]:
            if r["contact_id"] not in seen:
                seen.add(r["contact_id"])
                keep.append(r)
        st[names["colls"]["requests"]["name"]] = keep
    elif k == "cinema":
        movies = rng.sample(
            ["Lật Mặt 8", "Doraemon", "Địa Đạo", "Thám tử lừng danh Conan", "Mưa Đỏ", "Bộ Tứ Báo Thủ", "Cám"], 3
        )

        def sc():
            m = rng.choice(movies)
            t = future_slot(nowd, rng, 0, 3)
            return {
                "movie": m,
                "start": _iso(t),
                "cinema": rng.choice(["CGV Vincom", "Lotte Cinema", "BHD Star", "Galaxy"]),
                "price": rng.choice([85000, 95000, 110000]),
            }

        mk("showtimes", rng.randint(*dom.colls["showtimes"].n), sc)
        st[names["colls"]["showtimes"]["name"]].sort(key=lambda x: x["start"])
        coll("tickets")
    elif k == "building":

        def tk():
            return {
                "category": rng.choice(["electric", "water", "elevator"]),
                "description": rng.choice(
                    [
                        "đèn hành lang tầng năm không sáng",
                        "vòi nước bếp rỉ nước",
                        "thang máy số hai kêu to",
                        "ổ điện phòng khách bị chập",
                    ]
                ),
                "status": rng.choice(["open", "in_progress"]),
            }

        mk("tickets", rng.randint(*dom.colls["tickets"].n), tk)
        seen, keep = set(), []
        for x in st[names["colls"]["tickets"]["name"]]:
            if x["description"] not in seen:
                seen.add(x["description"])
                keep.append(x)
        st[names["colls"]["tickets"]["name"]] = keep
        coll("amenities")
    elif k == "clinic":
        ds = coll("doctors")

        def ap2():
            d = rng.choice(ds)
            t = future_slot(nowd, rng)
            return {
                "doctor_id": d["id"],
                "doctor_name": d["name"],
                "start": _iso(t),
                "end": _iso(t + timedelta(minutes=rng.choice([30, 45, 60]))),
                "status": "booked",
            }

        mk("appointments", rng.randint(*dom.colls["appointments"].n), ap2)
    if dom.status is not None:
        s = copy.deepcopy(dom.status)
        if k == "garden":
            s["zones"] = rng.sample(s["zones"], rng.randint(3, 5))
            s["tank_percent"] = rng.choice([12, 18, 25, 35, 48, 60, 75, 90])
            s["auto"] = rng.random() < 0.5
        elif k == "parking":
            s["price_per_month"] = rng.choice([120000, 150000, 200000, 300000])
            s["balance"] = rng.choice([80000, 150000, 250000, 400000, 600000, 1000000])
        elif k == "heater":
            s["power"] = rng.choice(["on", "off"])
            s["current_temp"] = rng.choice([28, 32, 41, 50, 55]) if s["power"] == "on" else rng.choice([26, 28, 30])
            s["target_temp"] = rng.choice([40, 45, 50, 55])
            s["mode"] = rng.choice(["eco", "boost"])
        st[dom.status_key] = s
    return st


# ---------------------------------------------------------------------------------------------------------------------
def app_class(zaic, dom: Domain, names: dict):
    """Lớp con zaic_tools.App: mỗi tool một method (đánh dấu @read/@write của zaic_tools)."""
    App, read, write = zaic.App, zaic.apps.read, zaic.apps.write
    app = names["app"]
    methods = {"name": app}

    def mkmethod(t: Tool):
        cm = names["colls"].get(t.coll) if t.coll else None
        pmap = {p.canon: pname(names, t, p) for p in t.params}

        def run(self, **kw):
            args = {c: kw.get(n) for c, n in pmap.items()}
            if t.op == "list":
                flt = {pmap[p.canon]: args[p.canon] for p in t.params if p.kind == "enum" and args[p.canon] is not None}
                return self.read(cm["name"], eq=flt or None)
            if t.op == "search":
                q = kw.get(names["params"][f"{t.key}.$q"])
                items = self.state.get(cm["name"], [])
                fields = tuple(
                    sorted(
                        {
                            k
                            for x in items
                            for k, v in x.items()
                            if k != "id" and isinstance(v, str) and not k.endswith(("_id", "start", "end"))
                        }
                    )
                )
                return self.read(cm["name"], contains={fields or (dom.colls[t.coll].key,): q})
            if t.op == "status":
                return {"ok": True, **{k: v for k, v in self.state[dom.status_key].items() if k != "watering"}}
            if t.op == "topup":
                st = self.state[dom.status_key]
                st["balance"] += args["amount"]
                return {"ok": True, "balance": st["balance"]}
            if t.op == "set" and dom.key == "parking":
                v = self.find(cm["name"], args["vehicle_id"])
                if v is None:
                    return self.noop(f"không tìm thấy {pmap['vehicle_id']}={args['vehicle_id']!r}")
                st = self.state[dom.status_key]
                cost = st["price_per_month"] * args["months"]
                if st["balance"] < cost:
                    return self.noop(f"số dư {st['balance']} không đủ {cost}")
                st["balance"] -= cost
                v["paid_months"] = v.get("paid_months", 0) + args["months"]
                return {"ok": True, "vehicle": v, "balance": st["balance"]}
            if t.op == "set":
                st = self.state[dom.status_key]
                if t.key == "water":
                    zone = next((z for z in st["zones"] if zaic.utils._eq(z, args["zone"])), None)
                    if zone is None:
                        return self.noop(f"không có khu tưới {args['zone']!r}")
                    st["watering"] = {"zone": zone, "minutes": args.get("minutes") or 10}
                    return {"ok": True, "watering": st["watering"]}
                if t.key == "auto":
                    st["auto"] = args["enabled"]
                    return {"ok": True, "auto": st["auto"]}
                if t.key == "stop":
                    if st.get("watering") is None:
                        return self.noop("không có vòi nào đang tưới")
                    st["watering"] = None
                    return {"ok": True}
                if t.key == "power":
                    if st["power"] == args["power"]:
                        return self.noop(f"bình đã {args['power']}")
                    st["power"] = args["power"]
                    return {"ok": True, "power": st["power"]}
                if t.key == "temp":
                    st.update(power="on", target_temp=args["temperature"])
                    if args.get("mode"):
                        st["mode"] = args["mode"]
                    return {"ok": True, **{k: st[k] for k in ("power", "target_temp", "mode")}}
                raise ValueError(t.key)
            # thao tác trên collection
            fields = {}
            for p in t.params:
                v = args[p.canon]
                if v is None:
                    continue
                if p.kind == "ref":
                    target = self.find(names["colls"][p.ref]["name"], v)
                    if target is None:
                        return self.noop(f"không tìm thấy {pmap[p.canon]}={v!r}")
                    if t.op == "create":
                        fields[p.canon] = v
                        label = {
                            "pets": "pet_name",
                            "vehicles": "vehicle_name",
                            "helpers": "helper_name",
                            "children": "child_name",
                            "products": "product_name",
                            "services": "service_name",
                            "doctors": "doctor_name",
                            "gifts": "gift_name",
                        }.get(p.ref)
                        if label:
                            fields[label] = target[dom.colls[p.ref].key]
                        if p.ref == "showtimes":
                            fields.update(movie=target["movie"], start=target["start"])
                        if p.ref == "services":
                            fields["minutes"] = target["minutes"]
                elif p.kind == "contact":
                    book = (self.session.states.get("contacts") or {}).get("contacts") or []
                    c = next((c for c in book if c.get("id") == v), None)
                    if c is None:
                        return self.noop(f"không tìm thấy {pmap[p.canon]}={v!r}")
                    fields["contact_id"] = v
                    fields["contact_name"] = c["name"]
                else:
                    fields[p.canon] = v
            if t.op == "create":
                cmn = names["colls"][t.coll]
                if dom.key == "clean":
                    s = datetime.fromisoformat(fields["start"])
                    fields["end"] = (s + timedelta(hours=fields["hours"])).isoformat()
                if dom.key == "salon":
                    s = datetime.fromisoformat(fields["start"])
                    fields["end"] = (s + timedelta(minutes=fields.pop("minutes"))).isoformat()
                if dom.key in ("pet", "clinic"):
                    s = datetime.fromisoformat(fields["start"])
                    fields["end"] = (s + timedelta(minutes=30)).isoformat()
                defaults = {
                    "status": "pending"
                    if dom.key in ("water", "split", "gift")
                    else "sent"
                    if dom.key == "school"
                    else "open"
                    if (dom.key, t.coll) == ("building", "tickets")
                    else "booked"
                }
                if dom.key == "split":
                    defaults["reminded"] = False
                out = self.create(cmn["name"], cmn["prefix"], defaults=defaults, **fields)
                return {"ok": True, "id": out["id"], cmn["sing"]: out["item"]}
            idp = next(p for p in t.params if p.kind == "ref" and p.ref == t.coll)
            item = self.find(cm["name"], args[idp.canon])
            if item.get("status") in ("cancelled", "delivered") or (
                t.sets and all(item.get(k) == v for k, v in t.sets.items())
            ):
                return self.noop(
                    f"{dom.colls[t.coll].noun} {item['id']} không đổi được (trạng thái {item.get('status')})"
                )
            if t.op == "update":
                upd = {k: v for k, v in fields.items() if k != idp.canon}
                return self.patch(cm["name"], pmap[idp.canon], item["id"], **upd)
            if t.op == "act":
                return self.patch(cm["name"], pmap[idp.canon], item["id"], set=dict(t.sets))
            raise ValueError(t.op)

        run.__name__ = names["tools"][t.key]
        return (read if t.op in ("list", "search", "status") else write)(run)

    for t in dom.tools:
        methods[names["tools"][t.key]] = mkmethod(t)
    return type(f"Synth_{app}", (App,), methods)


def install(session, zaic, metas: list[dict]) -> None:
    """Gắn các app tổng hợp vào Session THẬT của mẫu chủ: catalog, tool được phép, state, method; state_hash tính trên cả."""
    for m in metas:
        dom = BY_KEY[m["domain"]]
        cat = catalog(dom, m["names"])
        cls = app_class(zaic, dom, m["names"])
        app = m["names"]["app"]
        session.states[app] = copy.deepcopy(m["state"])
        session.apps[app] = cls(session, session.states[app])
        for t in cat["tools"]:
            session.tools[t["name"]] = t
            session.allowed.add(t["name"])
            zaic.apps.TOOLS[t["name"]] = t["kind"]


def make_env(zaic_mod_loader, root, host_id: str, metas: list[dict], installed_tools: list[str]):
    """ToolEnv thật của mẫu chủ + app tổng hợp. allowed = đúng tool đã cài trong mẫu tổng hợp."""
    zaic = zaic_mod_loader()
    s = zaic.ToolEnv("train", host_id, root=root)
    install(s, zaic, metas)
    s.allowed = set(installed_tools) | {"sys.ask_missing_info"}
    return s
