from abc import ABCMeta, abstractmethod
import unicodedata
import re
import math

def normalize(x):

    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')

    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)

    while True:
        
        old_x = x

        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        
        if x == old_x:
            break
    
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    
    return x


class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])

    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not math.isnan(amount) and not math.isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):

        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        
        self._year = year
        self._month = month
        self._day = day
        
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))

    __repr__ = __str__

    def match(self, other):
        
        assert isinstance(other, Value)
        
        if self.normalized == other.normalized:
            return True
        
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """

    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    
    if not corenlp_value:
        corenlp_value = original_string
    
    # Number?
    amount = NumberValue.parse(corenlp_value)
    
    if amount is not None:
        return NumberValue(amount, original_string)
    
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    
    # String.
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                        in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """

    # Check size
    if len(target_values) != len(predicted_values):
        return False
    
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    
    return True


def check_denotation_relaxed(
    target_values,
    predicted_values,
    numeric_abs_tol: float = 1e-3,
    numeric_rel_tol: float = 1e-3,
):
    """Order-insensitive denotation with numeric tolerance.

    This keeps the denotation matching behavior but accepts small numeric
    differences caused by formatting/rounding.
    """
    if len(target_values) != len(predicted_values):
        return False

    used = [False] * len(predicted_values)

    def _match_value(t, p):
        if t.match(p):
            return True
        if isinstance(t, NumberValue) and isinstance(p, NumberValue):
            diff = abs(t.amount - p.amount)
            scale = max(1.0, abs(t.amount), abs(p.amount))
            return (diff <= numeric_abs_tol) or (diff / scale <= numeric_rel_tol)
        return False

    for t in target_values:
        found = False
        for i, p in enumerate(predicted_values):
            if used[i]:
                continue
            if _match_value(t, p):
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True


def _to_items(ans):
    if ans is None:
        return []
    if isinstance(ans, (list, tuple, set)):
        return [str(x).strip() for x in ans if str(x).strip()]
    s = str(ans).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_numeric_text(s: str):
    t = str(s).strip().replace(",", "")
    if not t:
        return None
    try:
        v = float(t)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _all_numeric_items(items):
    vals = []
    for x in items:
        v = _parse_numeric_text(x)
        if v is None:
            return None
        vals.append(v)
    return vals


def _extract_numeric_values_from_text(text: str):
    s = str(text or "")
    # Keep grouping commas inside numbers, then normalize while parsing.
    nums = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", s)
    out = []
    for n in nums:
        v = _parse_numeric_text(n)
        if v is not None:
            out.append(v)
    return out


def _numeric_multiset_match(gold_nums, pred_nums, abs_tol: float, rel_tol: float):
    if len(gold_nums) != len(pred_nums):
        return False
    used = [False] * len(pred_nums)
    for g in gold_nums:
        found = False
        for i, p in enumerate(pred_nums):
            if used[i]:
                continue
            diff = abs(g - p)
            scale = max(1.0, abs(g), abs(p))
            if (diff <= abs_tol) or (diff / scale <= rel_tol):
                used[i] = True
                found = True
                break
        if not found:
            return False
    return True


def evaluate_prediction(
    gold_answers,
    pred_answer,
    numeric_abs_tol: float = 1e-3,
    numeric_rel_tol: float = 1e-3,
):
    """Evaluate one prediction with order-insensitive matching and numeric tolerance."""
    gold_items = _to_items(gold_answers)
    pred_text = "" if pred_answer is None else str(pred_answer).strip()

    # If there is a single gold span, try whole-span matching first to avoid
    # over-splitting entity answers that contain commas.
    pred_candidates = []
    if len(gold_items) == 1 and pred_text:
        pred_candidates.append([pred_text])
    pred_candidates.append(_to_items(pred_text))

    gold_values = to_value_list(gold_items)
    for pred_items in pred_candidates:
        pred_values = to_value_list(pred_items)
        if check_denotation_relaxed(
            gold_values,
            pred_values,
            numeric_abs_tol=numeric_abs_tol,
            numeric_rel_tol=numeric_rel_tol,
        ):
            return True

    # Composite-entity regrouping fallback:
    # If gold answers are multi-field entries (e.g., "name, school, city"),
    # regroup flattened prediction tokens by the same field width.
    if len(gold_items) > 1:
        gold_parts = [len([x.strip() for x in g.split(",") if x.strip()]) for g in gold_items]
        if gold_parts and min(gold_parts) == max(gold_parts) and gold_parts[0] > 1:
            k = gold_parts[0]
            pred_flat = _to_items(pred_text)
            if len(pred_flat) == len(gold_items) * k:
                regrouped = []
                for i in range(0, len(pred_flat), k):
                    regrouped.append(", ".join(pred_flat[i : i + k]))
                regrouped_values = to_value_list(regrouped)
                if check_denotation_relaxed(
                    gold_values,
                    regrouped_values,
                    numeric_abs_tol=numeric_abs_tol,
                    numeric_rel_tol=numeric_rel_tol,
                ):
                    return True

    # Numeric-list fallback: for questions whose gold answers are all numeric,
    # compare all numeric tokens in prediction text with tolerant multiset matching.
    gold_nums = _all_numeric_items(gold_items)
    if gold_nums is not None and pred_text:
        pred_nums = _extract_numeric_values_from_text(pred_text)
        if _numeric_multiset_match(
            gold_nums,
            pred_nums,
            abs_tol=numeric_abs_tol,
            rel_tol=numeric_rel_tol,
        ):
            return True

    # Single-entity fallback: allow "entity, count" style predictions
    # when gold expects one non-numeric entity.
    if len(gold_items) == 1:
        g = gold_items[0]
        if _parse_numeric_text(g) is None:
            g_val = to_value(g)
            for tok in _to_items(pred_text):
                if _parse_numeric_text(tok) is not None:
                    continue
                if g_val.match(to_value(tok)):
                    return True
    return False

def normalize_text(s: str) -> str:
    """
    Normalize a string by:
      1) Removing thousands separators inside numbers: 1,000 -> 1000 ; 12,345,678.9 -> 12345678.9
      2) Removing common unit suffixes from numbers: 1000m -> 1000 ; 1.5kg -> 1.5 ; 2,000 ms -> 2000
      3) Stripping Latin diacritics: Concepción, Chile -> Concepcion, Chile
    Everything else is left unchanged.
    """

    # 1) Remove commas used as thousands separators (but keep decimal parts)
    def _remove_grouping_commas(text: str) -> str:
        # Matches numbers like 1,234 or 12,345,678.9 and removes commas inside the number
        pattern = re.compile(r'(?<!\d)(\d{1,3}(?:,\d{3})+)(\.\d+)?')
        return pattern.sub(lambda m: m.group(1).replace(',', '') + (m.group(2) or ''), text)

    s = _remove_grouping_commas(s)

    # 2) Strip unit suffixes from numbers (whitelist of common SI/base units & variants)
    units = [
        # multi-char first to prefer longest match
        "µm", "um", "km", "cm", "mm", "ms", "kg", "mol", "min", "ml", "Pa", "Wh", "Hz",
        "hr", "hrs",
        # single/short forms
        "h", "m", "s", "g", "L", "l", "N", "J", "W", "K", "A", "d", "y"
    ]
    units.sort(key=len, reverse=True)
    units_alt = "|".join(re.escape(u) for u in units)
    # Matches: number (int/float, already de-commaed) + optional space + unit, e.g., "1000m", "1.5 kg", "2000 ms"
    unit_pattern = re.compile(rf'(?<!\w)(\d+(?:\.\d+)?)[ ]*({units_alt})(?!\w)')
    s = unit_pattern.sub(r"\1", s)

    # 3) Strip diacritics from Latin letters (e.g., "Concepción" -> "Concepcion")
    def _strip_diacritics(text: str) -> str:
        decomp = unicodedata.normalize("NFKD", text)
        # remove combining marks; then recompose to keep non-Latin scripts intact
        filtered = "".join(ch for ch in decomp if not unicodedata.combining(ch))
        return unicodedata.normalize("NFC", filtered)

    s = _strip_diacritics(s)

    return s

def calculate_acc(ground_truth, preds):
    count = 0
    for pred, gold in zip(preds, ground_truth):
        correct = evaluate_prediction(gold, pred)
        if correct:
            count += 1
    print(count, "Acc: ", count / len(preds))

def calculate_fact_acc(ground_truth, preds):
    count = 0
    length = len(preds)
    for i in range(length):
        if ground_truth[i].lower() == preds[i].lower():
            count += 1
    print(count, "Acc: ", count / length)

def strip_diacritics(text: str) -> str:
    decomp = unicodedata.normalize("NFKD", text)
    # remove combining marks; then recompose to keep non-Latin scripts intact
    filtered = "".join(ch for ch in decomp if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", filtered)