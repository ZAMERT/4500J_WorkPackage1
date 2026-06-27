import re
from typing import List, Tuple

CODE_PATTERNS = [
    r'^(LOCAL\s+)?(PERS|VAR|CONST)\b',
    r'^(LOCAL\s+)?PROC\b.*\)?$',
    r'^END(PROC|MODULE|TEST|IF|FOR|WHILE)$',
    r'^MODULE\b',
    r'^IF\b.*\bTHEN$',
    r'^(ELSE|ELSEIF\b.*\bTHEN)$',
    r'^FOR\b.*\bDO$',
    r'^WHILE\b.*\bDO$',
    r'^(ERROR|TRAP\b.*|TEST\b.*|CASE\b.*|DEFAULT:?)$',
    r'^(RETRY|TRYNEXT|RAISE|RETURN|CONNECT|IDelete|ISignalDI|StorePath|RestoPath|StartMove)\b.*;?$',
    r'^[A-Za-z_]\w*\s*:\s*=',  # assignment
    r'^[A-Za-z_]\w*(\s+|\\|\().*;',  # instruction/function call ending in semicolon
    r'^\.\.+;?$',  # omitted code marker in examples
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CODE_PATTERNS]


def is_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 5000:
        return False
    return any(p.search(stripped) for p in _COMPILED)


def find_code_blocks(text: str) -> List[Tuple[int, int]]:
    """
    返回文本中 RAPID 代码块的字符位置列表，每项为 (start, end)。
    连续的代码行合并为一个块；单行代码行之间若间隔不超过 1 个空行也会合并。
    """
    lines = text.split('\n')
    blocks: List[Tuple[int, int]] = []
    in_code = False
    block_start = 0
    blank_streak = 0       # 当前连续空行数
    MAX_BLANK_GAP = 1      # 允许代码块中间有几个空行
    pos = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for '\n'

        if is_code_line(line):
            if not in_code:
                in_code = True
                block_start = pos
            blank_streak = 0
        else:
            if in_code:
                if line.strip() == '':
                    blank_streak += 1
                    if blank_streak > MAX_BLANK_GAP:
                        # 空行太多，结束当前代码块
                        blocks.append((block_start, pos))
                        in_code = False
                        blank_streak = 0
                else:
                    # 非代码、非空行，结束代码块
                    blocks.append((block_start, pos))
                    in_code = False
                    blank_streak = 0

        pos += line_len

    # 文本结尾仍在代码块中
    if in_code:
        blocks.append((block_start, pos))

    return blocks


def spans_code_block(start: int, end: int, code_blocks: List[Tuple[int, int]]) -> bool:
    """
    判断 [start, end) 这个切割区间是否切入了某个代码块内部。
    即：切割点落在代码块中间（不是整块包含，而是只切了一部分）。
    """
    for cb_start, cb_end in code_blocks:
        # 切割区间与代码块有交叉，但没有完整包含代码块
        overlaps = start < cb_end and end > cb_start
        contains = start <= cb_start and end >= cb_end
        if overlaps and not contains:
            return True
    return False
