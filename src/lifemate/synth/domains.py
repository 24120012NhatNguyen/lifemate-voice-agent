"""Đặc tả các app tổng hợp: collection, trường, tool CRUD cộng vài động từ riêng, và mẫu câu tiếng Việt.

Không app nào trùng hay gần trùng app của train, của val_app (`splits.HELDOUT_APPS`) hay của public: val_app phải vẫn
đo app chưa thấy, và public vẫn là ước lượng thật. Mỗi lần dựng một app (`apps.instance`), tên app, tên collection,
tên tool và tên tham số được chọn ngẫu nhiên trong các phương án, nên mô hình phải đọc schema chứ không nhớ được tên.

Mẫu câu: `{p}` là đại từ người dùng tự xưng, `{obj}` là mục bị tác động (gọi theo `Coll.say`), `{<trường>}` là giá trị
của tham số đó nói ra. Đoạn trong `[...]` chỉ giữ khi tham số trong nó được nói; trường nói ra mà mẫu câu không có chỗ
thì mẫu đó không được chọn.

Tên tool có thể chứa `{coll}` và `{sing}` (tên collection và số ít của nó trong lần dựng đó).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Field:
    kind: str                       # text | enum | int | money | bool | date | datetime | ref | status
    names: tuple = ()               # tên thay thế của trường (cũng là tên tham số); rỗng: dùng tên chuẩn
    pool: tuple = ()                # text: giá trị mẫu
    values: dict | None = None      # enum, bool: {giá trị: [cách nói]}; status: các giá trị
    lo: int = 0                     # int, money: khoảng; date, datetime: khoảng ngày tính từ hôm nay
    hi: int = 0
    unit: tuple = ()                # int: đơn vị nói sau số ("người", "đêm")
    ref: str = ""                   # ref: tên chuẩn của collection đích
    desc: str = ""
    lower: bool = False             # text: người dùng nói chữ đầu viết thường; gold giữ chữ hoa đầu
    init: object = None             # status: giá trị lúc tạo


def text(pool=(), *names, desc="", lower=False):
    return Field("text", names, tuple(pool), desc=desc, lower=lower)


def enum(values, *names, desc=""):
    return Field("enum", names, values=values, desc=desc)


def integer(lo, hi, unit=(), *names, desc=""):
    return Field("int", names, lo=lo, hi=hi, unit=tuple(unit), desc=desc)


def money(lo, hi, *names, desc="Số tiền (VND)"):
    return Field("money", names, lo=lo, hi=hi, desc=desc)


def boolean(values, *names, desc=""):
    return Field("bool", names, values=values, desc=desc)


def date(lo, hi, *names, desc=""):
    return Field("date", names, lo=lo, hi=hi, desc=desc)


def dt(lo, hi, *names, desc=""):
    return Field("datetime", names, lo=lo, hi=hi, desc=desc)


def ref(coll, desc=""):
    return Field("ref", ref=coll, desc=desc)


def status(init, values, *names):
    return Field("status", names or ("status",), values={v: [] for v in values}, init=init)


@dataclass
class Coll:
    names: list                     # [(tên collection, số ít)]; số ít đặt tên tham số ID: <số ít>_id
    prefix: list                    # tiền tố ID
    say: list                       # cách gọi một mục: {key} nguyên văn, {lkey} chữ đầu viết thường
    noun: str                       # danh từ cho câu liệt kê ("thú cưng", "lịch hẹn")
    fields: dict
    pool: list                      # các mục mẫu (ít nhất có trường khoá)
    key: str | None = "name"        # trường làm tên gọi; None: lấy tên của mục cha qua `derived`
    derived: str | None = None      # trường ref mà tên gọi lấy theo mục cha
    n: tuple = (3, 6)               # số mục lúc đầu
    detail: tuple = ()              # trường chỉ tool get trả về (tool list bỏ đi)


@dataclass
class Tool:
    key: str                        # tên chuẩn, để `Domain.conds` trỏ tới
    op: str                         # list | search | get | create | update | delete | set | upsert
    coll: str
    names: list
    desc: str
    params: list = field(default_factory=list)   # tên trường chuẩn, "!" là bắt buộc (ID của op ghi tự thêm)
    say: list = field(default_factory=list)
    set: dict | None = None         # op set: {trường: giá trị}, "$today" là ngày hôm nay
    defaults: dict | None = None    # op create: giá trị mặc định
    by: str = ""                    # op upsert: trường để tìm mục đã có
    qnames: tuple = ("query", "keyword", "q")   # op search: tên tham số truy vấn


@dataclass
class Domain:
    ids: list
    vi: str
    group: str
    desc: list
    colls: dict
    tools: list
    conds: list = field(default_factory=list)  # [{coll, field, value, say, tool, fixed, slots}]


DT_DESC = "Thời điểm"
STATUS_BOOK = ["booked", "cancelled"]

DOMAINS = [
    Domain(
        ids=["pet_care", "pet_health"], vi="Chăm sóc thú cưng", group="personal",
        desc=["Hồ sơ thú cưng và lịch hẹn thú y", "Quản lý thú cưng, đặt lịch khám, tiêm, spa"],
        colls={
            "pets": Coll([("pets", "pet")], ["pet", "p"], ["bé {key}", "con {key}", "{key}"], "thú cưng",
                         fields={"name": text(desc="Tên thú cưng"),
                                 "species": enum({"dog": ["chó", "cún"], "cat": ["mèo"], "rabbit": ["thỏ"], "hamster": ["hamster"]},
                                                 "species", "kind", "type"),
                                 "weight_kg": integer(1, 35, ["ký", "kg"], "weight_kg", "weight", desc="Cân nặng (kg)")},
                         pool=[{"name": n} for n in ["Mích", "Lu", "Bông", "Mực", "Vàng", "Mít", "Kem", "Sữa", "Bơ", "Cốm",
                                                     "Mochi", "Su", "Tôm", "Gấu"]]),
            "visits": Coll([("visits", "visit"), ("appointments", "appointment")], ["visit", "appt"],
                           ["lịch {lkey}", "lịch hẹn {lkey}"], "lịch hẹn", key="title",
                           fields={"title": text(desc="Dịch vụ", lower=True), "pet": ref("pets"),
                                   "start": dt(0, 13, "start", "time", "at", desc=DT_DESC),
                                   "note": text(["bé hay cắn", "mang theo sổ tiêm", "bé đang bỏ ăn", "nhịn ăn trước 6 tiếng"],
                                                "note", desc="Ghi chú cho bác sĩ"),
                                   "status": status("booked", STATUS_BOOK)},
                           pool=[{"title": t} for t in ["Tiêm phòng dại", "Khám tổng quát", "Tẩy giun", "Cắt móng", "Tắm spa",
                                                        "Siêu âm", "Khám da", "Tiêm vắc xin 5 bệnh"]],
                           detail=("note",)),
        },
        tools=[
            Tool("list_pets", "list", "pets", ["list_{coll}", "get_{coll}"], "Danh sách thú cưng"),
            Tool("add_pet", "create", "pets", ["add_{sing}", "create_{sing}"], "Thêm hồ sơ thú cưng",
                 ["name!", "species!", "weight_kg"],
                 ["thêm[ bé {species}] {name} vào hồ sơ[, nặng {weight_kg}]",
                  "tạo hồ sơ cho {name}[, là {species}][, nặng {weight_kg}]",
                  "nhà mới nuôi thêm {name}[, bé là {species}], thêm vào app[, nặng {weight_kg}]"]),
            Tool("update_pet", "update", "pets", ["update_{sing}", "edit_{sing}"], "Sửa hồ sơ thú cưng",
                 ["weight_kg", "name"],
                 ["cập nhật cân nặng {obj}[ là {weight_kg}]", "{obj} giờ nặng {weight_kg} rồi, sửa hồ sơ giúp {p}",
                  "đổi tên {obj}[ thành {name}]"]),
            Tool("list_visits", "list", "visits", ["list_{coll}", "upcoming_{coll}"], "Các lịch hẹn thú y"),
            Tool("get_visit", "get", "visits", ["get_{sing}", "{sing}_detail"], "Chi tiết một lịch hẹn, kể cả ghi chú"),
            Tool("book_visit", "create", "visits", ["book_{sing}", "schedule_{sing}", "new_{sing}"], "Đặt lịch hẹn thú y",
                 ["pet!", "title!", "start!", "note"],
                 ["đặt lịch[ {title}][ cho {pet}][ {start}][, ghi chú {note}]",
                  "[{start} ]đưa {pet} đi[ {title}], đặt lịch giúp {p}[, ghi chú là {note}]",
                  "book lịch[ {title}][ cho {pet}][ vào {start}]"]),
            Tool("move_visit", "update", "visits", ["reschedule_{sing}", "move_{sing}"], "Dời lịch hẹn", ["start!"],
                 ["dời {obj}[ sang {start}]", "đổi {obj}[ qua {start}]", "chuyển {obj}[ sang {start}] giúp {p}"]),
            Tool("cancel_visit", "set", "visits", ["cancel_{sing}"], "Huỷ lịch hẹn",
                 say=["huỷ {obj}", "hủy {obj} giúp {p}", "{obj} không đi được nữa, huỷ đi"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["plant_care", "garden"], vi="Chăm cây", group="home",
        desc=["Cây trong nhà và lịch tưới", "Theo dõi cây cảnh, nhắc tưới"],
        colls={
            "plants": Coll([("plants", "plant")], ["plant", "pl"], ["cây {key}", "chậu {key}"], "cây",
                           fields={"name": text(desc="Tên cây"),
                                   "location": text(["ban công", "phòng khách", "bàn làm việc", "sân thượng", "cửa sổ bếp"],
                                                    "location", desc="Chỗ đặt cây"),
                                   "water_every_days": integer(1, 14, ["ngày"], "water_every_days", "interval_days",
                                                               desc="Mấy ngày tưới một lần"),
                                   "last_watered": date(-6, -1, "last_watered", desc="Ngày tưới gần nhất")},
                           pool=[{"name": n} for n in ["Kim tiền", "Lưỡi hổ", "Trầu bà", "Sen đá", "Lan hồ điệp", "Xương rồng",
                                                       "Cau tiểu trâm", "Bàng Singapore", "Phát tài", "Ngọc ngân", "Vạn niên thanh",
                                                       "Hoa giấy", "Húng quế", "Bạc hà"]]),
        },
        tools=[
            Tool("list_plants", "list", "plants", ["list_{coll}", "{coll}"], "Danh sách cây và lịch tưới"),
            Tool("add_plant", "create", "plants", ["add_{sing}", "new_{sing}"], "Thêm cây",
                 ["name!", "water_every_days!", "location"],
                 ["thêm cây {name}[ ở {location}] vào danh sách[, tưới {water_every_days} một lần]",
                  "{p} mới mua chậu {name}[, để {location}], thêm vào giúp {p}[, {water_every_days} tưới một lần]",
                  "ghi thêm cây {name}[, cứ {water_every_days} tưới một lần][, đặt ở {location}]"]),
            Tool("water", "set", "plants", ["water_{sing}", "mark_watered"], "Ghi đã tưới hôm nay",
                 say=["đánh dấu đã tưới {obj}", "{p} vừa tưới {obj} xong, ghi lại giúp {p}", "ghi là {obj} tưới rồi nha"],
                 set={"last_watered": "$today"}),
            Tool("set_interval", "update", "plants", ["set_schedule", "update_{sing}"], "Đổi lịch tưới",
                 ["water_every_days!"],
                 ["đổi lịch tưới {obj}[ thành {water_every_days} một lần]", "{obj} giờ tưới {water_every_days} một lần thôi"]),
            Tool("remove_plant", "delete", "plants", ["remove_{sing}", "delete_{sing}"], "Xoá cây khỏi danh sách",
                 say=["xoá {obj} khỏi danh sách", "bỏ {obj} đi, cây chết rồi", "xoá giúp {p} {obj}"]),
        ]),
    Domain(
        ids=["car_care", "vehicle_service"], vi="Bảo dưỡng xe", group="transport",
        desc=["Xe của bạn, số km và lịch bảo dưỡng", "Sổ bảo dưỡng xe, đặt lịch garage"],
        colls={
            "vehicles": Coll([("vehicles", "vehicle"), ("cars", "car")], ["veh", "car"], ["xe {key}", "chiếc {key}"], "xe",
                             fields={"name": text(desc="Tên xe"), "odometer_km": integer(1200, 90000, ["km", "cây số"], "odometer_km",
                                                                                        "mileage", desc="Số km đã chạy")},
                             pool=[{"name": n} for n in ["Vios", "SH", "Vision", "Air Blade", "CR-V", "Morning", "Wave", "Lead",
                                                         "VF 5", "Exciter", "City"]], n=(2, 4)),
            "services": Coll([("services", "service"), ("bookings", "booking")], ["svc", "bk"],
                             ["lịch {lkey}", "lịch {lkey} xe"], "lịch bảo dưỡng", key="title",
                             fields={"title": text(desc="Việc cần làm", lower=True), "vehicle": ref("vehicles"),
                                     "start": dt(0, 13, "start", "time", desc=DT_DESC),
                                     "status": status("booked", STATUS_BOOK)},
                             pool=[{"title": t} for t in ["Thay nhớt", "Bảo dưỡng định kỳ", "Rửa xe", "Thay lốp", "Đăng kiểm",
                                                          "Vệ sinh nồi", "Thay má phanh", "Kiểm tra điều hoà"]]),
        },
        tools=[
            Tool("list_vehicles", "list", "vehicles", ["list_{coll}", "my_{coll}"], "Danh sách xe"),
            Tool("set_km", "update", "vehicles", ["update_odometer", "set_mileage"], "Cập nhật số km", ["odometer_km!"],
                 ["cập nhật số km {obj}[ là {odometer_km}]", "{obj} chạy được {odometer_km} rồi, cập nhật giúp {p}"]),
            Tool("list_services", "list", "services", ["list_{coll}", "upcoming_{coll}"], "Lịch bảo dưỡng đã đặt"),
            Tool("book_service", "create", "services", ["book_{sing}", "schedule_{sing}"], "Đặt lịch bảo dưỡng",
                 ["vehicle!", "title!", "start!"],
                 ["đặt lịch[ {title}][ cho {vehicle}][ {start}]", "[{start} ]mang {vehicle} đi[ {title}], đặt lịch trước giúp {p}",
                  "book garage[ {title}][ cho {vehicle}][ lúc {start}]"]),
            Tool("cancel_service", "set", "services", ["cancel_{sing}"], "Huỷ lịch bảo dưỡng",
                 say=["huỷ {obj}", "thôi huỷ {obj} đi"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["restaurant_booking", "table_reservation"], vi="Đặt bàn nhà hàng", group="food",
        desc=["Tìm nhà hàng và đặt bàn", "Đặt chỗ nhà hàng, quán ăn"],
        colls={
            "restaurants": Coll([("restaurants", "restaurant")], ["rst", "res"], ["nhà hàng {key}", "quán {key}", "{key}"],
                                "nhà hàng",
                                fields={"name": text(desc="Tên nhà hàng"),
                                        "cuisine": text(["lẩu", "nướng", "Nhật", "chay", "hải sản", "Ý"], "cuisine")},
                                pool=[{"name": n} for n in ["Bếp Nhà Mây", "Lẩu Gió Bấc", "Sushi Kaze", "Cơm Niêu Phố Cũ",
                                                            "Pizza Lửa Hồng", "Quán Gánh", "Nướng Bếp Than", "Chay Tịnh Tâm",
                                                            "Hải Sản Bờ Kè", "Lẩu Nấm Rừng Xanh"]]),
            "reservations": Coll([("reservations", "reservation"), ("bookings", "booking")], ["rsv", "bk"],
                                 ["bàn ở {key}", "chỗ đặt ở {key}", "lịch đặt bàn ở {key}"], "bàn đã đặt", key=None,
                                 derived="restaurant",
                                 fields={"restaurant": ref("restaurants"), "start": dt(0, 13, "start", "time", "at", desc=DT_DESC),
                                         "party_size": integer(1, 12, ["người", "khách"], "party_size", "guests", "people",
                                                               desc="Số người"),
                                         "note": text(["bàn gần cửa sổ", "có ghế trẻ em", "phòng riêng", "tổ chức sinh nhật"],
                                                      "note", desc="Yêu cầu thêm"),
                                         "status": status("booked", STATUS_BOOK)},
                                 pool=[{}], n=(1, 3)),
        },
        tools=[
            Tool("search_restaurants", "search", "restaurants", ["search_{coll}", "find_{sing}"],
                 "Tìm nhà hàng theo tên; không có query thì trả mọi nhà hàng"),
            Tool("reserve", "create", "reservations", ["reserve_table", "book_table", "create_{sing}"], "Đặt bàn",
                 ["restaurant!", "start!", "party_size!", "note"],
                 ["đặt bàn ở {restaurant}[ {start}][ cho {party_size}][, {note}]",
                  "book cho {p} một bàn[ {party_size}] ở {restaurant}[ {start}][, ghi chú {note}]",
                  "[{start} ]{p} muốn ăn ở {restaurant}[, {party_size}], đặt bàn trước giúp {p}"]),
            Tool("list_reservations", "list", "reservations", ["list_{coll}", "my_{coll}"], "Các bàn đã đặt"),
            Tool("change_reservation", "update", "reservations", ["update_{sing}", "change_{sing}"], "Đổi giờ hoặc số người",
                 ["start", "party_size"],
                 ["đổi {obj}[ sang {start}][ thành {party_size}]", "dời {obj}[ qua {start}]",
                  "{obj} đi thêm người, sửa thành {party_size}"]),
            Tool("cancel_reservation", "set", "reservations", ["cancel_{sing}"], "Huỷ bàn đã đặt",
                 say=["huỷ {obj}", "không đi nữa, huỷ {obj} giúp {p}"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["hotel_booking", "stays"], vi="Đặt phòng khách sạn", group="travel",
        desc=["Tìm khách sạn, đặt và quản lý phòng", "Đặt phòng lưu trú"],
        colls={
            "hotels": Coll([("hotels", "hotel")], ["htl", "h"], ["khách sạn {key}", "{key}"], "khách sạn",
                           fields={"name": text(desc="Tên khách sạn"),
                                   "price_per_night": money(400_000, 3_500_000, "price_per_night", "price")},
                           pool=[{"name": n} for n in ["Hoa Sen Riverside", "Biển Xanh Resort", "Phố Núi Lodge", "Mây Trắng",
                                                       "Cát Vàng Boutique", "Đồi Thông Villa", "Sao Mai", "Hải Âu Inn"]]),
            "bookings": Coll([("bookings", "booking"), ("stays", "stay")], ["bk", "stay"],
                             ["phòng ở {key}", "phòng đặt ở {key}", "lịch ở {key}"], "phòng đã đặt", key=None, derived="hotel",
                             fields={"hotel": ref("hotels"), "check_in": date(1, 13, "check_in", "checkin_date", desc="Ngày nhận phòng"),
                                     "nights": integer(1, 7, ["đêm"], "nights", desc="Số đêm"),
                                     "guests": integer(1, 6, ["người", "khách"], "guests", "adults", desc="Số khách"),
                                     "status": status("booked", STATUS_BOOK)},
                             pool=[{}], n=(1, 2)),
        },
        tools=[
            Tool("search_hotels", "search", "hotels", ["search_{coll}", "find_{sing}"],
                 "Tìm khách sạn theo tên; không có query thì trả mọi khách sạn"),
            Tool("book_room", "create", "bookings", ["book_room", "create_{sing}", "reserve_room"], "Đặt phòng",
                 ["hotel!", "check_in!", "nights!", "guests"],
                 ["đặt phòng ở {hotel}[ từ {check_in}][ {nights}][ cho {guests}]",
                  "book {hotel}[ {nights}][, nhận phòng {check_in}][, {guests}]",
                  "[{check_in} ]{p} đi chơi, đặt giúp {p} phòng ở {hotel}[ {nights}]"]),
            Tool("list_bookings", "list", "bookings", ["list_{coll}", "my_{coll}"], "Các phòng đã đặt"),
            Tool("change_nights", "update", "bookings", ["extend_{sing}", "update_{sing}"], "Đổi số đêm", ["nights!"],
                 ["đổi {obj}[ thành {nights}]", "{p} ở thêm, sửa {obj}[ thành {nights}]"]),
            Tool("cancel_booking", "set", "bookings", ["cancel_{sing}"], "Huỷ đặt phòng",
                 say=["huỷ {obj}", "huỷ giúp {p} {obj}, lịch thay đổi rồi"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["beauty_salon", "salon_booking", "spa"], vi="Làm đẹp, spa", group="personal",
        desc=["Dịch vụ salon, spa và lịch hẹn", "Đặt lịch cắt tóc, làm móng, spa"],
        colls={
            "services": Coll([("services", "service"), ("treatments", "treatment")], ["sv", "trt"],
                             ["dịch vụ {lkey}", "gói {lkey}", "{lkey}"], "dịch vụ",
                             fields={"name": text(desc="Tên dịch vụ", lower=True),
                                     "duration_minutes": integer(20, 120, ["phút"], "duration_minutes", "minutes"),
                                     "price": money(80_000, 900_000, "price")},
                             pool=[{"name": n} for n in ["Cắt tóc nam", "Gội dưỡng sinh", "Nhuộm tóc", "Uốn tóc", "Làm móng tay",
                                                         "Massage cổ vai gáy", "Chăm sóc da mặt", "Tẩy da chết"]]),
            "appointments": Coll([("appointments", "appointment"), ("bookings", "booking")], ["apt", "bk"],
                                 ["lịch {key}", "lịch hẹn {key}"], "lịch hẹn", key=None, derived="service",
                                 fields={"service": ref("services"), "start": dt(0, 13, "start", "time", desc=DT_DESC),
                                         "status": status("booked", STATUS_BOOK)},
                                 pool=[{}], n=(1, 3)),
        },
        tools=[
            Tool("list_services", "list", "services", ["list_{coll}", "{coll}"], "Các dịch vụ và giá"),
            Tool("book", "create", "appointments", ["book_{sing}", "book"], "Đặt lịch hẹn", ["service!", "start!"],
                 ["đặt lịch {service}[ {start}]", "hẹn {service}[ {start}] giúp {p}", "[{start} ]{p} muốn làm {service}, đặt lịch giúp {p}"]),
            Tool("list_appointments", "list", "appointments", ["list_{coll}", "my_{coll}"], "Lịch hẹn đã đặt"),
            Tool("reschedule", "update", "appointments", ["reschedule_{sing}", "move_{sing}"], "Dời lịch hẹn", ["start!"],
                 ["dời {obj}[ sang {start}]", "đổi {obj}[ qua {start}] giúp {p}"]),
            Tool("cancel", "set", "appointments", ["cancel_{sing}"], "Huỷ lịch hẹn",
                 say=["huỷ {obj}", "{p} bận rồi, huỷ {obj} đi"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["home_repair", "handyman"], vi="Thợ sửa chữa tại nhà", group="home",
        desc=["Gọi thợ điện, nước, máy lạnh đến nhà", "Yêu cầu sửa chữa nhà cửa"],
        colls={
            "requests": Coll([("requests", "request"), ("jobs", "job"), ("tickets", "ticket")], ["req", "job"],
                             ["yêu cầu {lkey}", "vụ {lkey}", "lịch thợ {lkey}"], "yêu cầu sửa chữa", key="title",
                             fields={"title": text(desc="Việc cần sửa", lower=True),
                                     "category": enum({"electric": ["điện"], "plumbing": ["nước", "điện nước"],
                                                       "aircon": ["máy lạnh", "điều hoà"], "painting": ["sơn"]},
                                                      "category", "type"),
                                     "start": dt(0, 13, "start", "visit_time", "time", desc="Giờ thợ đến"),
                                     "address": text(["nhà riêng", "căn hộ tầng 12", "nhà ba mẹ", "văn phòng"], "address"),
                                     "stars": integer(1, 5, ["sao"], "stars", "rating", desc="Số sao 1-5"),
                                     "status": status("open", ["open", "cancelled", "done"])},
                             pool=[{"title": t} for t in ["Sửa vòi nước rò rỉ", "Thay bóng đèn hành lang", "Vệ sinh máy lạnh",
                                                          "Thông bồn rửa chén", "Sửa ổ điện phòng ngủ", "Sơn lại tường phòng khách",
                                                          "Thay ống nước nhà tắm", "Sửa cửa cuốn"]]),
        },
        tools=[
            Tool("list_requests", "list", "requests", ["list_{coll}", "my_{coll}"], "Các yêu cầu sửa chữa"),
            Tool("create_request", "create", "requests", ["create_{sing}", "new_{sing}", "book_technician"], "Gọi thợ",
                 ["title!", "category!", "start!", "address"],
                 ["đặt thợ[ {category}][ {start}] qua {title}[ ở {address}]",
                  "gọi thợ[ {category}] tới {title}[ {start}]", "[{start} ]cho thợ qua {title}[, bên {category}]"]),
            Tool("move_request", "update", "requests", ["reschedule_{sing}", "update_{sing}"], "Đổi giờ thợ đến", ["start!"],
                 ["dời {obj}[ sang {start}]", "đổi giờ {obj}[ thành {start}]"]),
            Tool("cancel_request", "set", "requests", ["cancel_{sing}"], "Huỷ yêu cầu",
                 say=["huỷ {obj}", "{p} tự làm rồi, huỷ {obj} đi"], set={"status": "cancelled"}),
            Tool("rate", "update", "requests", ["rate_{sing}", "review_{sing}"], "Chấm điểm thợ sau khi làm xong",
                 ["stars!"], ["chấm {obj}[ {stars}]", "đánh giá {obj}[ {stars}] nha, thợ làm tốt"]),
        ]),
    Domain(
        ids=["expense_tracker", "spending"], vi="Sổ chi tiêu", group="finance",
        desc=["Ghi chi tiêu và ngân sách theo mục", "Theo dõi chi tiêu cá nhân"],
        colls={
            "expenses": Coll([("expenses", "expense"), ("entries", "entry")], ["exp", "e"],
                             ["khoản {lkey}", "khoản chi {lkey}", "mục {lkey}"], "khoản chi", key="title",
                             fields={"title": text(desc="Chi cho việc gì", lower=True),
                                     "amount": money(20_000, 2_500_000, "amount"),
                                     "category": enum({"food": ["ăn uống"], "transport": ["đi lại"], "shopping": ["mua sắm"],
                                                       "bills": ["hoá đơn"], "fun": ["giải trí"]}, "category"),
                                     "date": date(-1, 0, "date", "spent_on", desc="Ngày chi")},
                             pool=[{"title": t} for t in ["Ăn trưa", "Đổ xăng", "Cà phê", "Tiền điện", "Mua sách", "Đi chợ",
                                                          "Taxi", "Xem phim", "Mua áo khoác", "Gửi xe", "Ăn sáng"]]),
            "budgets": Coll([("budgets", "budget")], ["bud", "b"], ["ngân sách {key}"], "ngân sách", key="category",
                            fields={"category": enum({"food": ["ăn uống"], "transport": ["đi lại"], "shopping": ["mua sắm"],
                                                      "bills": ["hoá đơn"], "fun": ["giải trí"]}, "category"),
                                    "limit": money(500_000, 8_000_000, "limit", "monthly_limit", desc="Hạn mức mỗi tháng (VND)")},
                            pool=[], n=(0, 0)),
        },
        tools=[
            Tool("list_expenses", "list", "expenses", ["list_{coll}", "{coll}"], "Các khoản đã ghi"),
            Tool("add_expense", "create", "expenses", ["add_{sing}", "log_{sing}"], "Ghi một khoản chi",
                 ["title!", "amount!", "category", "date"],
                 ["ghi {title}[ {amount}][ {date}]", "thêm khoản chi {title}[ hết {amount}][, mục {category}][ {date}]",
                  "[{date} ]{p} {title} hết {amount}, ghi vào sổ giúp {p}[, mục {category}]"]),
            Tool("edit_expense", "update", "expenses", ["update_{sing}", "edit_{sing}"], "Sửa khoản chi", ["amount", "category"],
                 ["sửa {obj}[ thành {amount}]", "{obj} ghi nhầm, đổi lại {amount}", "chuyển {obj} sang mục {category}"]),
            Tool("delete_expense", "delete", "expenses", ["delete_{sing}", "remove_{sing}"], "Xoá khoản chi",
                 say=["xoá {obj} đi", "bỏ {obj} ra khỏi sổ, ghi trùng rồi"]),
            Tool("list_budgets", "list", "budgets", ["list_{coll}", "get_{coll}"], "Hạn mức chi theo mục"),
            Tool("set_budget", "upsert", "budgets", ["set_budget", "update_budget"], "Đặt hạn mức chi theo mục",
                 ["category!", "limit!"], ["đặt ngân sách {category}[ tháng này là {limit}]",
                                           "giới hạn chi {category}[ {limit}] một tháng thôi"], by="category"),
        ]),
    Domain(
        ids=["watchlist", "movie_list"], vi="Phim muốn xem", group="entertainment",
        desc=["Danh sách phim muốn xem và đánh giá", "Lưu phim, đánh dấu đã xem"],
        colls={
            "titles": Coll([("titles", "title"), ("movies", "movie")], ["mv", "t"], ["phim {key}", "{key}"], "phim",
                           key="name",
                           fields={"name": text(desc="Tên phim"),
                                   "kind": enum({"movie": ["phim lẻ"], "series": ["phim bộ"]}, "kind", "type", "format"),
                                   "watched": boolean({True: ["đã xem"], False: ["chưa xem"]}, "watched"),
                                   "stars": integer(1, 5, ["sao"], "stars", "rating", desc="Đánh giá 1-5 sao")},
                           pool=[{"name": n} for n in ["Mắt Biếc", "Bố Già", "Nhà Bà Nữ", "Hai Phượng", "Inception",
                                                       "Interstellar", "Dune", "Tấm Cám", "Em Chưa 18", "Vây Hãm",
                                                       "Chị Chị Em Em", "Lật Mặt"]]),
        },
        tools=[
            Tool("list_titles", "list", "titles", ["list_{coll}", "get_{coll}"], "Danh sách phim đã lưu"),
            Tool("add_title", "create", "titles", ["add_{sing}", "save_{sing}"], "Lưu phim muốn xem", ["name!", "kind"],
                 ["lưu phim {name}[, {kind}] vào danh sách xem sau", "thêm {name}[, {kind}] vào danh sách phim",
                  "nhớ giùm {p} phim {name} để coi sau"], defaults={"watched": False}),
            Tool("mark_watched", "set", "titles", ["mark_watched", "set_watched"], "Đánh dấu đã xem",
                 say=["đánh dấu {obj} là đã xem", "{p} coi xong {obj} rồi"], set={"watched": True}),
            Tool("rate_title", "update", "titles", ["rate_{sing}", "review_{sing}"], "Chấm sao cho phim", ["stars!"],
                 ["chấm {obj}[ {stars}]", "{obj} hay đó, cho {stars}"]),
            Tool("remove_title", "delete", "titles", ["remove_{sing}", "delete_{sing}"], "Xoá phim khỏi danh sách",
                 say=["xoá {obj} khỏi danh sách", "bỏ {obj} đi, không muốn xem nữa"]),
        ],
        conds=[dict(coll="titles", field="watched", value=True, tool="remove_title",
                    say=["nếu {p} coi {obj} rồi thì xoá khỏi danh sách", "{obj} mà đã xem rồi thì bỏ ra khỏi danh sách"])]),
    Domain(
        ids=["photo_album", "gallery"], vi="Album ảnh", group="personal",
        desc=["Album ảnh trên điện thoại", "Quản lý album, chia sẻ ảnh"],
        colls={
            "albums": Coll([("albums", "album")], ["alb", "a"], ["album {key}"], "album",
                           fields={"name": text(desc="Tên album"),
                                   "shared": boolean({True: ["đang chia sẻ"], False: ["riêng tư"]}, "shared"),
                                   "pinned": boolean({True: ["đã ghim"], False: ["chưa ghim"]}, "pinned")},
                           pool=[{"name": n} for n in ["Đà Lạt 2025", "Sinh nhật bé Na", "Tết Bính Ngọ", "Đám cưới anh Hai",
                                                       "Du lịch Phú Quốc", "Họp lớp cấp 3", "Trung thu", "Ảnh thẻ"]]),
        },
        tools=[
            Tool("list_albums", "list", "albums", ["list_{coll}", "{coll}"], "Các album"),
            Tool("create_album", "create", "albums", ["create_{sing}", "new_{sing}"], "Tạo album", ["name!"],
                 ["tạo album {name}", "tạo giúp {p} một album tên là {name}", "làm một album mới, đặt tên {name}"],
                 defaults={"shared": False, "pinned": False}),
            Tool("rename_album", "update", "albums", ["rename_{sing}", "update_{sing}"], "Đổi tên album", ["name!"],
                 ["đổi tên {obj}[ thành {name}]", "sửa tên {obj}[ lại là {name}]"]),
            Tool("share_album", "set", "albums", ["share_{sing}", "enable_sharing"], "Bật chia sẻ album",
                 say=["chia sẻ {obj}", "bật chia sẻ {obj} cho cả nhà xem"], set={"shared": True}),
            Tool("pin_album", "set", "albums", ["pin_{sing}"], "Ghim album lên đầu",
                 say=["ghim {obj} lên đầu", "đưa {obj} lên đầu danh sách"], set={"pinned": True}),
            Tool("delete_album", "delete", "albums", ["delete_{sing}", "remove_{sing}"], "Xoá album",
                 say=["xoá {obj}", "xoá luôn {obj} đi"]),
        ]),
    Domain(
        ids=["apartment_services", "building_portal"], vi="Dịch vụ chung cư", group="home",
        desc=["Đăng ký khách đến chơi, đặt tiện ích toà nhà", "Cổng cư dân chung cư"],
        colls={
            "visitors": Coll([("visitors", "visitor"), ("guests", "guest")], ["vis", "g"], ["khách {key}", "{key}"],
                             "khách đã đăng ký",
                             fields={"name": text(desc="Tên khách"), "start": dt(0, 6, "arrival", "start", "time", desc="Giờ khách đến"),
                                     "status": status("registered", ["registered", "cancelled"])},
                             pool=[{"name": n} for n in ["anh Tuấn", "chị Lan", "bác Hùng", "cô Hoa", "anh Phúc", "chú Bảy",
                                                         "em Nga", "anh Khoa"]], n=(1, 3)),
            "amenities": Coll([("amenities", "amenity"), ("facilities", "facility")], ["am", "fac"], ["{lkey}", "khu {lkey}"],
                              "tiện ích",
                              fields={"name": text(desc="Tên tiện ích", lower=True)},
                              pool=[{"name": n} for n in ["Bể bơi", "Sân tennis", "Phòng sinh hoạt cộng đồng", "Khu BBQ",
                                                          "Sân cầu lông", "Phòng đọc sách"]]),
            "amenity_bookings": Coll([("amenity_bookings", "amenity_booking"), ("slots", "slot")], ["ab", "slot"],
                                     ["lịch {key}", "giờ đặt {key}"], "giờ đã đặt", key=None, derived="amenity",
                                     fields={"amenity": ref("amenities"), "start": dt(0, 6, "start", "time", desc=DT_DESC),
                                             "hours": integer(1, 3, ["tiếng"], "hours", desc="Số giờ"),
                                             "status": status("booked", STATUS_BOOK)},
                                     pool=[{}], n=(0, 2)),
        },
        tools=[
            Tool("register_visitor", "create", "visitors", ["register_{sing}", "add_{sing}"], "Báo bảo vệ có khách đến",
                 ["name!", "start!"],
                 ["đăng ký khách {name}[ đến {start}]", "báo bảo vệ[ {start}] có {name} tới chơi",
                  "[{start} ]{name} qua nhà, đăng ký khách giúp {p}"]),
            Tool("list_visitors", "list", "visitors", ["list_{coll}"], "Khách đã đăng ký"),
            Tool("cancel_visitor", "set", "visitors", ["cancel_{sing}"], "Huỷ đăng ký khách",
                 say=["huỷ đăng ký {obj}", "{obj} không qua nữa, huỷ giúp {p}"], set={"status": "cancelled"}),
            Tool("list_amenities", "list", "amenities", ["list_{coll}"], "Tiện ích của toà nhà"),
            Tool("book_amenity", "create", "amenity_bookings", ["book_{sing}", "book_amenity"], "Đặt giờ dùng tiện ích",
                 ["amenity!", "start!", "hours"],
                 ["đặt {amenity}[ {start}][ {hours}]", "giữ giúp {p} {amenity}[ {start}][, {hours}]"]),
            Tool("list_amenity_bookings", "list", "amenity_bookings", ["list_{coll}", "my_{coll}"], "Giờ tiện ích đã đặt"),
            Tool("cancel_amenity", "set", "amenity_bookings", ["cancel_{sing}"], "Huỷ giờ đã đặt",
                 say=["huỷ {obj}", "thôi không dùng nữa, huỷ {obj}"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["trip_planner", "itinerary"], vi="Lịch trình du lịch", group="travel",
        desc=["Lên lịch trình chuyến đi và các điểm đến", "Kế hoạch du lịch"],
        colls={
            "trips": Coll([("trips", "trip")], ["trip", "tr"], ["chuyến {key}", "lịch trình {key}"], "chuyến đi",
                          fields={"name": text(desc="Tên chuyến"), "start_date": date(1, 8, "start_date", "from_date", desc="Ngày đi"),
                                  "end_date": date(9, 13, "end_date", "to_date", desc="Ngày về")},
                          pool=[{"name": n} for n in ["Đà Lạt cuối tuần", "Hội An", "Sapa mùa lúa", "Về quê", "Phú Quốc nghỉ dưỡng",
                                                      "Hà Giang", "Quy Nhơn", "Côn Đảo"]], n=(1, 3)),
            "places": Coll([("places", "place"), ("stops", "stop")], ["pl", "stop"], ["{lkey}", "điểm {lkey}"], "điểm đến",
                           fields={"name": text(desc="Tên điểm đến", lower=True), "trip": ref("trips"),
                                   "date": date(1, 13, "date", "visit_date", desc="Ngày ghé")},
                           pool=[{"name": n} for n in ["Chợ đêm", "Phố đi bộ", "Bãi biển phía nam", "Quán cà phê trên đồi",
                                                       "Bảo tàng", "Làng gốm", "Thác nước", "Chợ hải sản"]], n=(1, 4)),
        },
        tools=[
            Tool("list_trips", "list", "trips", ["list_{coll}", "my_{coll}"], "Các chuyến đi"),
            Tool("create_trip", "create", "trips", ["create_{sing}", "new_{sing}"], "Tạo chuyến đi",
                 ["name!", "start_date!", "end_date!"],
                 ["tạo chuyến {name}[ từ {start_date}][ tới {end_date}]", "lên lịch trình {name}[, đi {start_date}][, về {end_date}]"]),
            Tool("add_place", "create", "places", ["add_{sing}", "add_stop"], "Thêm điểm đến vào chuyến", ["trip!", "name!", "date"],
                 ["thêm {name} vào {trip}[, ghé {date}]", "{trip} cho {p} ghé thêm {name}[ {date}]"]),
            Tool("list_places", "list", "places", ["list_{coll}"], "Các điểm đến của mọi chuyến"),
            Tool("remove_place", "delete", "places", ["remove_{sing}", "delete_{sing}"], "Bỏ một điểm đến",
                 say=["bỏ {obj} ra khỏi lịch trình", "xoá {obj} đi, không kịp ghé"]),
            Tool("delete_trip", "delete", "trips", ["delete_{sing}", "cancel_{sing}"], "Xoá chuyến đi",
                 say=["xoá {obj}", "huỷ {obj}, không đi nữa"]),
        ]),
    Domain(
        ids=["home_wifi", "router"], vi="Wifi gia đình", group="home",
        desc=["Thiết bị đang dùng wifi nhà, chặn và giới hạn tốc độ", "Quản lý router"],
        colls={
            "devices": Coll([("devices", "device"), ("clients", "client")], ["dev", "cl"], ["{lkey}"], "thiết bị",
                            fields={"name": text(desc="Tên thiết bị", lower=True),
                                    "blocked": boolean({True: ["đang bị chặn"], False: ["đang dùng mạng"]}, "blocked"),
                                    "limit_mbps": integer(2, 100, ["Mbps", "mê"], "limit_mbps", "speed_limit", desc="Tốc độ tối đa (Mbps)")},
                            pool=[{"name": n} for n in ["Điện thoại của Bin", "Laptop bố", "iPad của bé Na", "Tivi phòng khách",
                                                        "Máy chơi game", "Camera cổng", "Laptop công ty", "Điện thoại mẹ"]]),
        },
        tools=[
            Tool("list_devices", "list", "devices", ["list_{coll}", "connected_{coll}"], "Thiết bị đang kết nối"),
            Tool("block", "set", "devices", ["block_{sing}", "pause_{sing}"], "Chặn mạng một thiết bị",
                 say=["chặn mạng {obj}", "ngắt wifi {obj} đi", "khoá mạng {obj} giúp {p}"], set={"blocked": True}),
            Tool("unblock", "set", "devices", ["unblock_{sing}", "resume_{sing}"], "Mở lại mạng cho thiết bị",
                 say=["mở mạng lại cho {obj}", "bỏ chặn {obj}"], set={"blocked": False}),
            Tool("set_limit", "update", "devices", ["set_speed_limit", "limit_{sing}"], "Giới hạn tốc độ", ["limit_mbps!"],
                 ["giới hạn tốc độ {obj}[ còn {limit_mbps}]", "{obj} xài mạng nhiều quá, hạ xuống {limit_mbps} thôi"]),
        ],
        conds=[dict(coll="devices", field="blocked", value=False, tool="block",
                    say=["{obj} mà còn đang dùng mạng thì chặn luôn", "nếu {obj} chưa bị chặn thì chặn đi"])]),
    Domain(
        ids=["warranty", "warranty_tracker"], vi="Sổ bảo hành", group="personal",
        desc=["Đồ đã mua, hạn bảo hành và yêu cầu bảo hành", "Theo dõi bảo hành sản phẩm"],
        colls={
            "products": Coll([("products", "product"), ("items", "item")], ["prd", "it"], ["{lkey}", "cái {lkey}"], "đồ đã mua",
                             fields={"name": text(desc="Tên sản phẩm", lower=True),
                                     "warranty_months": integer(6, 36, ["tháng"], "warranty_months", "months", desc="Số tháng bảo hành"),
                                     "purchase_date": date(-1, 0, "purchase_date", "bought_on", desc="Ngày mua")},
                             pool=[{"name": n} for n in ["Tủ lạnh Toshiba", "Laptop Dell", "Điện thoại Samsung", "Nồi chiên không dầu",
                                                         "Máy lọc nước", "Tivi Sony", "Máy hút bụi", "Quạt điều hoà"]]),
            "claims": Coll([("claims", "claim"), ("tickets", "ticket")], ["clm", "tk"], ["yêu cầu bảo hành {key}"],
                           "yêu cầu bảo hành", key=None, derived="product",
                           fields={"product": ref("products"),
                                   "note": text(["không lên nguồn", "kêu to", "rò nước", "màn hình bị sọc", "sạc không vào"],
                                                "note", "content", desc="Mô tả lỗi"),
                                   "status": status("open", ["open", "cancelled"])},
                           pool=[{}], n=(0, 2)),
        },
        tools=[
            Tool("list_products", "list", "products", ["list_{coll}", "{coll}"], "Đồ đã mua và hạn bảo hành"),
            Tool("add_product", "create", "products", ["add_{sing}", "register_{sing}"], "Lưu sản phẩm mới mua",
                 ["name!", "warranty_months!", "purchase_date"],
                 ["thêm {name} vào sổ bảo hành[, bảo hành {warranty_months}][, mua {purchase_date}]",
                  "[{purchase_date} ]{p} mới mua {name}[, bảo hành {warranty_months}], lưu lại giúp {p}"]),
            Tool("file_claim", "create", "claims", ["file_{sing}", "create_{sing}"], "Gửi yêu cầu bảo hành",
                 ["product!", "note!"],
                 ["gửi yêu cầu bảo hành cho {product}[, lỗi {note}]", "{product} bị {note}, báo bảo hành giúp {p}"]),
            Tool("list_claims", "list", "claims", ["list_{coll}"], "Yêu cầu bảo hành đã gửi"),
            Tool("cancel_claim", "set", "claims", ["cancel_{sing}", "withdraw_{sing}"], "Rút yêu cầu bảo hành",
                 say=["huỷ {obj}", "rút {obj} lại, máy chạy lại rồi"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["insurance", "my_insurance"], vi="Bảo hiểm", group="finance",
        desc=["Hợp đồng bảo hiểm, gia hạn và yêu cầu bồi thường", "Quản lý bảo hiểm cá nhân"],
        colls={
            "policies": Coll([("policies", "policy"), ("contracts", "contract")], ["pol", "ct"], ["{lkey}", "gói {lkey}"],
                             "hợp đồng bảo hiểm",
                             fields={"name": text(desc="Tên hợp đồng", lower=True),
                                     "auto_renew": boolean({True: ["bật"], False: ["tắt"]}, "auto_renew", "enabled",
                                                           desc="Tự động gia hạn"),
                                     "expires": date(5, 13, "expires", "end_date", desc="Ngày hết hạn")},
                             pool=[{"name": n} for n in ["Bảo hiểm xe máy", "Bảo hiểm sức khoẻ gia đình", "Bảo hiểm nhà",
                                                         "Bảo hiểm du lịch", "Bảo hiểm ô tô", "Bảo hiểm tai nạn"]], n=(2, 4)),
            "claims": Coll([("claims", "claim")], ["clm", "c"], ["yêu cầu bồi thường {key}", "hồ sơ bồi thường {key}"],
                           "yêu cầu bồi thường", key=None, derived="policy",
                           fields={"policy": ref("policies"), "amount": money(300_000, 9_000_000, "amount"),
                                   "note": text(["viện phí", "va quẹt xe", "mất hành lý", "nước tràn vào nhà"], "note",
                                                desc="Lý do"),
                                   "status": status("submitted", ["submitted", "withdrawn"])},
                           pool=[{}], n=(0, 2)),
        },
        tools=[
            Tool("list_policies", "list", "policies", ["list_{coll}", "my_{coll}"], "Các hợp đồng bảo hiểm"),
            Tool("set_renew", "update", "policies", ["set_auto_renew", "toggle_renewal"], "Bật hoặc tắt tự động gia hạn",
                 ["auto_renew!"], ["{auto_renew} tự động gia hạn cho {obj}", "{auto_renew} gia hạn tự động {obj} giúp {p}"]),
            Tool("submit_claim", "create", "claims", ["submit_{sing}", "file_{sing}"], "Gửi yêu cầu bồi thường",
                 ["policy!", "amount!", "note"],
                 ["gửi yêu cầu bồi thường {policy}[ {amount}][, {note}]", "làm hồ sơ bồi thường {policy}[ cho khoản {note}][ {amount}]"]),
            Tool("list_claims", "list", "claims", ["list_{coll}"], "Yêu cầu bồi thường đã gửi"),
            Tool("withdraw_claim", "set", "claims", ["withdraw_{sing}", "cancel_{sing}"], "Rút yêu cầu bồi thường",
                 say=["rút {obj}", "huỷ {obj} giúp {p}"], set={"status": "withdrawn"}),
        ],
        conds=[dict(coll="policies", field="auto_renew", value=False, tool="set_renew", fixed={"auto_renew": True},
                    say=["nếu {obj} chưa bật tự gia hạn thì bật lên", "{obj} mà đang tắt gia hạn tự động thì bật giúp {p}"])]),
    Domain(
        ids=["court_booking", "sports_booking"], vi="Đặt sân thể thao", group="entertainment",
        desc=["Đặt sân cầu lông, tennis, bóng đá theo giờ", "Đặt sân thể thao"],
        colls={
            "courts": Coll([("courts", "court"), ("venues", "venue")], ["crt", "v"], ["{lkey}"], "sân",
                           fields={"name": text(desc="Tên sân", lower=True),
                                   "sport": enum({"badminton": ["cầu lông"], "tennis": ["tennis"], "football": ["bóng đá"],
                                                  "pickleball": ["pickleball"]}, "sport"),
                                   "price_per_hour": money(80_000, 600_000, "price_per_hour", "price")},
                           pool=[{"name": n} for n in ["Sân cầu lông Bình Minh", "Sân tennis Lá Phong", "Sân bóng Đồng Xanh",
                                                       "Sân pickleball Ngói Đỏ", "Sân cầu lông Hướng Dương", "Sân bóng Cầu Vồng"]]),
            "bookings": Coll([("bookings", "booking"), ("reservations", "reservation")], ["bk", "rsv"],
                             ["giờ đặt ở {key}", "lịch đá ở {key}", "lịch {key}"], "giờ sân đã đặt", key=None, derived="court",
                             fields={"court": ref("courts"), "start": dt(0, 13, "start", "time", desc=DT_DESC),
                                     "hours": integer(1, 3, ["tiếng"], "hours", "duration_hours", desc="Số giờ"),
                                     "status": status("booked", STATUS_BOOK)},
                             pool=[{}], n=(0, 2)),
        },
        tools=[
            Tool("list_courts", "list", "courts", ["list_{coll}", "{coll}"], "Các sân và giá theo giờ"),
            Tool("book_court", "create", "bookings", ["book_{sing}", "book_court", "reserve"], "Đặt sân",
                 ["court!", "start!", "hours!"],
                 ["đặt {court}[ {start}][, {hours}]", "giữ {court}[ {hours}][ {start}] giúp {p}",
                  "[{start} ]{p} đi đánh ở {court}, đặt sân trước[ {hours}]"]),
            Tool("list_bookings", "list", "bookings", ["list_{coll}", "my_{coll}"], "Giờ sân đã đặt"),
            Tool("move_booking", "update", "bookings", ["change_{sing}", "reschedule_{sing}"], "Đổi giờ đặt sân", ["start!"],
                 ["dời {obj}[ sang {start}]", "đổi {obj}[ qua {start}]"]),
            Tool("cancel_booking", "set", "bookings", ["cancel_{sing}"], "Huỷ giờ đã đặt",
                 say=["huỷ {obj}", "mưa rồi, huỷ {obj} giúp {p}"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["charity", "donations"], vi="Quyên góp từ thiện", group="finance",
        desc=["Chương trình từ thiện và các khoản ủng hộ", "Ủng hộ quỹ từ thiện"],
        colls={
            "campaigns": Coll([("campaigns", "campaign"), ("funds", "fund")], ["cmp", "f"], ["quỹ {key}", "chương trình {key}"],
                              "chương trình",
                              fields={"name": text(desc="Tên chương trình")},
                              pool=[{"name": n} for n in ["Áo ấm vùng cao", "Bữa cơm 0 đồng", "Cầu đá nông thôn", "Sách cho em",
                                                          "Nước sạch miền Tây", "Mái ấm Hoa Hồng"]]),
            "donations": Coll([("donations", "donation"), ("pledges", "pledge")], ["dn", "pl"],
                              ["khoản ủng hộ {key}", "khoản góp {key}"], "khoản ủng hộ", key=None, derived="campaign",
                              fields={"campaign": ref("campaigns"), "amount": money(50_000, 3_000_000, "amount"),
                                      "monthly": boolean({True: ["hàng tháng", "mỗi tháng"], False: ["một lần"]}, "monthly", "recurring",
                                                         desc="Ủng hộ hàng tháng"),
                                      "status": status("active", ["active", "cancelled"])},
                              pool=[{}], n=(0, 2)),
        },
        tools=[
            Tool("list_campaigns", "list", "campaigns", ["list_{coll}", "{coll}"], "Các chương trình đang gây quỹ"),
            Tool("donate", "create", "donations", ["donate", "create_{sing}"], "Ủng hộ một chương trình",
                 ["campaign!", "amount!", "monthly"],
                 ["ủng hộ {campaign}[ {amount}][ {monthly}]", "góp {amount} cho {campaign}[, {monthly}]",
                  "chuyển[ {amount}] ủng hộ {campaign}[ {monthly}]"]),
            Tool("list_donations", "list", "donations", ["list_{coll}", "my_{coll}"], "Các khoản đã ủng hộ"),
            Tool("stop_donation", "set", "donations", ["cancel_{sing}", "stop_{sing}"], "Dừng khoản ủng hộ",
                 say=["dừng {obj}", "huỷ {obj} giúp {p}"], set={"status": "cancelled"}),
        ]),
    Domain(
        ids=["gas_water_delivery", "home_delivery"], vi="Gọi gas, nước", group="home",
        desc=["Đặt bình gas, nước uống giao tận nhà", "Giao gas và nước bình"],
        colls={
            "products": Coll([("products", "product"), ("items", "item")], ["prd", "it"], ["{lkey}"], "mặt hàng",
                             fields={"name": text(desc="Tên mặt hàng", lower=True), "price": money(30_000, 600_000, "price"),
                                     "in_stock": boolean({True: ["còn hàng"], False: ["hết hàng"]}, "in_stock", "available")},
                             pool=[{"name": n} for n in ["Bình gas 12 ký", "Bình gas 45 ký", "Bình nước 19 lít",
                                                         "Thùng nước suối 24 chai", "Than hoa 5 ký", "Bình gas mini"]]),
            "orders": Coll([("orders", "order")], ["ord", "o"], ["đơn {key}", "đơn đặt {key}"], "đơn hàng", key=None,
                           derived="product",
                           fields={"product": ref("products"), "quantity": integer(1, 4, (), "quantity", "qty", desc="Số lượng"),
                                   "start": dt(0, 3, "deliver_at", "delivery_time", desc="Giờ giao"),
                                   "status": status("placed", ["placed", "cancelled"])},
                           pool=[{}], n=(0, 2)),
        },
        tools=[
            Tool("list_products", "list", "products", ["list_{coll}", "{coll}"], "Mặt hàng, giá và tình trạng còn hàng"),
            Tool("order", "create", "orders", ["place_{sing}", "order"], "Đặt hàng giao tận nhà",
                 ["product!", "quantity!", "start"],
                 ["đặt {quantity} {product}[ giao {start}]", "gọi {quantity} {product}[, giao {start}]",
                  "lấy giúp {p} {quantity} {product}[ {start}]"]),
            Tool("list_orders", "list", "orders", ["list_{coll}", "my_{coll}"], "Đơn đã đặt"),
            Tool("cancel_order", "set", "orders", ["cancel_{sing}"], "Huỷ đơn", say=["huỷ {obj}", "nhà còn rồi, huỷ {obj} đi"],
                 set={"status": "cancelled"}),
        ],
        conds=[dict(coll="products", field="in_stock", value=True, tool="order", slots=["quantity"],
                    say=["nếu {obj} còn hàng thì đặt {quantity}, không thì thôi", "{obj} còn thì lấy {p} {quantity}, hết thì thôi"])]),
    Domain(
        ids=["school_contact", "parent_portal"], vi="Liên lạc nhà trường", group="family",
        desc=["Xin phép nghỉ học cho con", "Cổng phụ huynh: con, đơn xin nghỉ"],
        colls={
            "children": Coll([("children", "child"), ("students", "student")], ["kid", "st"], ["bé {key}", "con {key}", "{key}"],
                             "con",
                             fields={"name": text(desc="Tên con"), "grade": text(["1A", "2B", "3C", "4A", "5D"], "grade", "class_name")},
                             pool=[{"name": n} for n in ["Bin", "Na", "Tôm", "Bống", "Su", "Gạo", "Bi", "Cún", "Mây"]], n=(1, 3)),
            "leaves": Coll([("leave_requests", "leave_request"), ("absences", "absence")], ["lv", "ab"],
                           ["đơn nghỉ của {key}", "đơn xin nghỉ cho {key}"], "đơn xin nghỉ", key=None, derived="child",
                           fields={"child": ref("children"), "date": date(0, 13, "date", "day", desc="Ngày nghỉ"),
                                   "note": text(["bị sốt", "đi khám răng", "về quê", "bị ho", "đi đám cưới"], "note", "content",
                                                desc="Lý do"),
                                   "status": status("sent", ["sent", "cancelled"])},
                           pool=[{}], n=(0, 2)),
        },
        tools=[
            Tool("list_children", "list", "children", ["list_{coll}", "my_{coll}"], "Các con đang học"),
            Tool("request_leave", "create", "leaves", ["request_leave", "create_{sing}"], "Xin phép nghỉ học",
                 ["child!", "date!", "note"],
                 ["xin cho {child} nghỉ học[ {date}][ vì {note}]", "báo cô[ {date}] {child} nghỉ[, lý do {note}]",
                  "{child}[ {note}], xin phép cô cho nghỉ[ {date}]"]),
            Tool("list_leaves", "list", "leaves", ["list_{coll}"], "Đơn xin nghỉ đã gửi"),
            Tool("cancel_leave", "set", "leaves", ["cancel_{sing}"], "Rút đơn xin nghỉ",
                 say=["rút {obj}", "{p} cho đi học lại rồi, huỷ {obj}"], set={"status": "cancelled"}),
        ]),
]
