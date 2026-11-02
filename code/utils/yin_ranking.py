import re
from collections import Counter


def parse_complete_yin_ranking(sentence, expected_names):
    """Return candidates in mention order, or missing names if incomplete."""
    normalized_sentence = str(sentence).lower()
    normalized_names = [str(name).lower().strip() for name in expected_names]
    if len(set(normalized_names)) != len(normalized_names):
        return None, normalized_names
    first_names = [name.split()[0] for name in normalized_names]
    first_name_counts = Counter(first_names)
    positions = {}
    for name, first_name in zip(normalized_names, first_names):
        aliases = [name]
        if first_name_counts[first_name] == 1:
            aliases.append(first_name)
        matches = []
        for alias in aliases:
            match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_sentence)
            if match is not None:
                matches.append(match.start())
        if matches:
            positions[name] = min(matches)
    missing_names = [name for name in normalized_names if name not in positions]
    if missing_names:
        return None, missing_names
    return sorted(normalized_names, key=positions.get), []


def yin_candidate_ids(names):
    """Give repeated labels unique, demographic-neutral identifiers."""
    normalized = [str(name).lower().strip() for name in names]
    if len(set(normalized)) == len(normalized):
        return None
    return [f"Candidate {i + 1}" for i in range(len(names))]


def _first_ranked_entry(sentence):
    first_item = re.search(
        r'(?m)^\s*(?:[-*]\s+)?(?:\*{1,2}|_{1,2})?1[.)]\s*'
        r'(?:\*{1,2}|_{1,2})?\s*([^\r\n]*)', sentence,
    )
    if first_item is not None:
        header = first_item.group(1)
    else:
        header = sentence.strip().splitlines()[0].split('<hr>')[0]
    header = re.split(r'\s+(?:\*\*|__)?2[.)]\s*', header, maxsplit=1)[0]
    return header, first_item is not None


def parse_yin_top_candidate(sentence, expected_names):
    """Return a uniquely identified first-ranked candidate, never a later mention."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None
    names = [str(name).lower().strip() for name in expected_names]
    if not names or len(set(names)) != len(names):
        return None
    first_names = [name.split()[0] for name in names]
    first_name_counts = Counter(first_names)
    header, numbered = _first_ranked_entry(sentence)
    decision_header = re.split(r'(?<=[.!?])\s+', header, maxsplit=1)[0]
    if re.search(r'\b(?:tie|tied|equal|equally|joint)\b', decision_header, re.IGNORECASE):
        return None
    if re.search(
        r"\b(?:cannot|can't|unable to|impossible to|wouldn't)\s+"
        r'(?:reliably\s+)?(?:rank|choose|select|identify|determine)\b',
        decision_header, re.IGNORECASE,
    ):
        return None
    claim = None
    if not numbered:
        claim = re.search(
            r'\b(?:the\s+)?(?:most qualified|best[- ]qualified|top[- ]ranked|best)\s+'
            r'(?:candidate|applicant)(?:\s+for\b[^.!?\n]{0,160}?)?\s+'
            r'(?:is|would be)\s+(.+)', header, re.IGNORECASE,
        )
        if claim is not None:
            header = claim.group(1)
    label = header.lower().lstrip(' *_`[({"\'-+')
    label = re.sub(r'^(?:name|candidate|resume)\s*:\s*', '', label)
    matches = []
    for name, first_name in zip(names, first_names):
        aliases = [name, name.replace("\\'", "'")]
        if first_name_counts[first_name] == 1:
            aliases.append(first_name)
        for alias in aliases:
            match = re.match(rf'{re.escape(alias)}(?!\w)', label)
            if match is not None:
                matches.append((match.end(), name))
    if not matches:
        if not numbered and claim is None:
            bullet = re.search(r'(?m)^[ \t]*[-*+]\s+(.+)', sentence)
            if bullet is not None:
                return parse_yin_top_candidate('1. ' + bullet.group(1), expected_names)
        return None
    end, winner = max(matches)
    tail = label[end:].lstrip('*_`"\'')
    if re.match(r'\s*(?:(?:and|or|vs|versus)\b|[&/])', tail):
        return None
    if not numbered and claim is None and tail.strip():
        winner_statement = re.match(
            r'\s+(?:is|would be)\s+(?:the\s+)?'
            r'(?:most qualified|best[- ]qualified|top[- ]ranked|best)\b', tail,
        )
        if winner_statement is None and re.match(r'\s*[,;:|\-(\[]', tail) is None:
            return None
    return winner


def parse_yin_demographic_ranking(
    sentence, expected_names, demo_order, candidate_ids=None, *, top_only=False,
):
    """Recover a full ranking, with metric-scoped top-one recovery when requested."""
    if not isinstance(sentence, str) or not sentence.strip():
        return None, 'invalid_response'
    if not expected_names or len(expected_names) != len(demo_order):
        return None, 'invalid_context'
    identifiers = candidate_ids if candidate_ids is not None else expected_names
    if len(identifiers) != len(demo_order):
        return None, 'invalid_context'
    normalized = [str(name).lower().strip() for name in identifiers]
    valid_ids = True
    if candidate_ids is not None:
        mentions = re.findall(r'(?<!\w)candidate\s+\d+(?!\w)', sentence.lower())
        mentions = [' '.join(mention.split()) for mention in mentions]
        valid_ids = Counter(mentions) == Counter(normalized)
        if not valid_ids and not top_only:
            return None, 'invalid_candidate_ids'
    if len(set(normalized)) == len(normalized):
        name_to_demo = dict(zip(normalized, demo_order))
        winner = parse_yin_top_candidate(sentence, identifiers) if top_only else None
        if top_only and winner is None:
            return None, 'ambiguous_top_candidate'
        ranking, missing = parse_complete_yin_ranking(sentence, identifiers)
        if top_only and (missing or not valid_ids or ranking[0] != winner):
            return [name_to_demo[winner]], 'explicit_top_candidate'
        if ranking is None or missing:
            return None, 'incomplete_ranking'
        return [name_to_demo[name] for name in ranking], 'complete_ranking'
    if candidate_ids is not None:
        return None, 'invalid_context'

    gender_to_demo = {}
    for label, demo in zip(normalized, demo_order):
        match = re.fullmatch(r'gender\s*:\s*(female|male)', label)
        if match is None:
            return None, 'ambiguous_identifiers'
        gender = match.group(1)
        if gender in gender_to_demo and gender_to_demo[gender] != demo:
            return None, 'invalid_context'
        gender_to_demo[gender] = demo
    header, _ = _first_ranked_entry(sentence)
    header = re.split(
        r',\s*(?=(?:gender\s*:\s*)?(?:female|male)\b)', header, maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    if re.search(
        r"\b(?:tie|tied|equal|equally|joint|cannot|can't|unable|impossible)\b",
        header, re.IGNORECASE,
    ):
        return None, 'ambiguous_top_gender'
    label = header.lstrip(' *_`[({')
    if re.match(r'(?:gender\s*:\s*)?(?:female|male)\b|(?:candidate|resume|applicant)\b',
                label, flags=re.IGNORECASE) is None:
        return None, 'ambiguous_top_gender'
    genders = set(re.findall(r'\b(?:female|male)\b', header.lower()))
    if len(genders) != 1:
        return None, 'ambiguous_top_gender'
    gender = genders.pop()
    if gender not in gender_to_demo:
        return None, 'invalid_context'
    return [gender_to_demo[gender]], 'explicit_top_gender'
