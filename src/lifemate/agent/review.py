"""LM-97: bounded, opt-in review of a proposal before it reaches the environment.

Only the messages already visible to the agent are used. No labels, simulator clone,
additional observations, app-specific rules, or generated training data are consumed.
"""
from __future__ import annotations

import json

from .. import data, splits

ASK = 'sys.ask_missing_info'

COMMON = (
    'Hành động JSON vừa nêu chỉ là bản nháp, CHƯA được thực thi. '
    'Kiểm tra lại bằng yêu cầu cuối, lịch sử, kết quả tool đã thấy và schema. '
    'Không coi suy đoán trong bản nháp là dữ kiện. '
    'Trả đúng một JSON hành động kế tiếp theo schema, thought ngắn nêu dữ kiện quyết định. '
    'Nếu bản nháp đúng thì giữ nguyên tên tool và toàn bộ arguments. '
    'Có thể giữ stop hoặc hỏi lại khi thật sự thiếu thông tin; không ép thực hiện. '
    'Không bịa ID/giá trị, không đọc lại dữ kiện đã đủ, không lặp việc đã hoàn thành. '
)
PROMPTS = {
    'source': COMMON + (
        'Rà nguồn: tool được chọn có đúng nghiệp vụ người dùng yêu cầu không? '
        'Phân biệt nguồn đọc cung cấp đối tượng cần thao tác với nguồn có tên gần giống. '
        'ID đã xuất hiện chưa đủ: phải thuộc đúng loại đối tượng/x-ref của tham số. '
        'Trước khi hỏi hoặc dừng, kiểm tra dữ kiện trong lịch sử và kết quả đã đọc; '
        'nếu còn thiếu nhưng có tool đọc phù hợp thì chọn bước đọc đó với bộ lọc từ lời người dùng. '
        'Nếu nhiều đối tượng vẫn hợp lệ và không có nguồn phân biệt thì hỏi lại, không tự chọn.'
    ),
    'verify': COMMON + (
        'Rà điều kiện và giá trị: xác định đúng đối tượng theo mọi điều kiện của người dùng, '
        'bao gồm câu tự sửa, phủ định, sớm/muộn nhất và điều kiện nếu/thì. '
        'So điều kiện với giá trị thực trong lịch sử/kết quả đọc trước khi ghi. '
        'Với ngày giờ/số lượng: kiểm tra nguồn của mốc gốc, đơn vị, phép cộng/trừ/nhân và giá trị đích; '
        'không nhầm giá trị hiện tại với ngưỡng đã đặt, giờ bắt đầu với giờ kết thúc. '
        'Trường optional vẫn cần nếu người dùng đã yêu cầu. '
        'Nếu chưa có dữ kiện để kiểm điều kiện, chọn đọc nguồn thích hợp; '
        'nếu điều kiện sai thì bỏ hành động đó, chỉ làm phần còn lại có căn cứ hoặc dừng.'
    ),
}


class InferenceReview:
    """Configuration: enabled=false, mode=source|verify, max_reviews=2, known=train_rest.

    Scope is samples with at least one installed non-sys app outside `known`, not
    oracle intent. source reviews the first proposal and later ask/stop proposals;
    verify reviews write/stop proposals. Neither reviews after an accepted ask.
    Counters live in ReactJSON.run, not on this shared agent (workers may be >1).
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        if not isinstance(cfg, dict):
            raise ValueError('agent.inference_review must be a mapping')
        self.enabled = cfg.get('enabled', False)
        if not isinstance(self.enabled, bool):
            raise ValueError('inference_review.enabled must be boolean')
        self.mode = cfg.get('mode', 'source')
        self.max_reviews = cfg.get('max_reviews', 2)
        self.known = cfg.get('known', 'train_rest')
        if self.mode not in PROMPTS:
            raise ValueError('inference_review.mode must be source or verify')
        if type(self.max_reviews) is not int or not 0 <= self.max_reviews <= 2:
            raise ValueError('inference_review.max_reviews must be 0, 1, or 2')
        if not (isinstance(self.known, list) and all(isinstance(a, str) for a in self.known)
                or self.known in ('train_rest', 'all')):
            raise ValueError('inference_review.known must be train_rest, all, or an app list')

    def sample_enabled(self, sample):
        if not self.enabled or not self.max_reviews:
            return False
        known = set(self.known) if isinstance(self.known, list) else splits.known_apps(self.known)
        return any(app != 'sys' and app not in known and a.get('tools')
                   for app, a in sample['context']['installed_apps'].items())

    def eligible(self, split, act, steps, used):
        if used >= self.max_reviews or any(s.get('name') == ASK for s in steps):
            return False
        if act.get('action') == 'stop':
            return True
        if act.get('action') != 'call':
            return False
        if self.mode == 'source':
            return not steps or act.get('name') == ASK
        return data.tool_specs(split).get(act.get('name'), {}).get('kind') == 'write'

    def messages(self, msgs, raw):
        return [*msgs, {'role': 'assistant', 'content': raw},
                {'role': 'user', 'content': PROMPTS[self.mode]}]


def valid_action(act, request):
    """Check output even if a server ignores JSON constraints; preserve existing masks."""
    props = request['response_format']['json_schema']['schema']['properties']
    if not isinstance(act, dict) or act.get('action') not in props['action']['enum']:
        return False
    if act['action'] == 'stop':
        return True
    return (act.get('name') in props['name']['enum']
            and isinstance(act.get('arguments'), dict))


def action_key(act):
    """Ignore changed explanations / JSON key ordering when counting replacements."""
    value = {'action': act['action']}
    if act['action'] == 'call':
        value.update(name=act.get('name'), arguments=act.get('arguments', {}))
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
