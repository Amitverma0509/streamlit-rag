"""Line-by-line port of textSplitter.js (RecursiveCharacterTextSplitter re-implementation)."""
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

def _len(s):  # JS String.length counts UTF-16 code units
    return len(s.encode("utf-16-le")) // 2

def _split_on_separator(text, separator):
    if separator == "":
        return list(text)
    return text.split(separator)

def _merge_splits(splits, separator, chunk_size, chunk_overlap):
    chunks = []
    current_chunk = []
    current_len = 0
    sep_len = _len(separator)

    for split in splits:
        split_len = _len(split)

        if current_len + split_len + (sep_len if current_chunk else 0) > chunk_size and current_chunk:
            joined = separator.join(current_chunk)
            if joined.strip():
                chunks.append(joined)

            # Build overlap: keep trailing pieces whose combined length is <= chunk_overlap
            while (
                current_len > chunk_overlap
                or (current_len + split_len + sep_len > chunk_size and current_len > 0)
            ):
                current_len -= _len(current_chunk[0]) + (sep_len if len(current_chunk) > 1 else 0)
                current_chunk.pop(0)

        current_chunk.append(split)
        current_len += split_len + (sep_len if len(current_chunk) > 1 else 0)

    if current_chunk:
        joined = separator.join(current_chunk)
        if joined.strip():
            chunks.append(joined)

    return chunks

def _recursive_split(text, separators, chunk_size, chunk_overlap):
    final_chunks = []
    separator = separators[-1]
    remaining_separators = []

    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    splits = [s for s in _split_on_separator(text, separator) if s != ""]
    good_splits = []

    for s in splits:
        if _len(s) < chunk_size:
            good_splits.append(s)
        else:
            if good_splits:
                final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))
                good_splits = []
            if not remaining_separators:
                final_chunks.append(s)
            else:
                final_chunks.extend(_recursive_split(s, remaining_separators, chunk_size, chunk_overlap))

    if good_splits:
        final_chunks.extend(_merge_splits(good_splits, separator, chunk_size, chunk_overlap))

    return final_chunks

def split_text(text, chunk_size=1000, chunk_overlap=200, separators=DEFAULT_SEPARATORS):
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    return [c for c in (c.strip() for c in _recursive_split(cleaned, separators, chunk_size, chunk_overlap)) if c]
