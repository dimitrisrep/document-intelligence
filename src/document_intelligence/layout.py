"""Assemble annotation regions into physical rows without using entity labels."""

from statistics import median

from document_intelligence.contracts import Line


def assemble_rows(lines):
    rows = []
    for line in sorted(
        lines, key=lambda x: (x.page, (x.bbox[1] + x.bbox[3]) / 2, x.bbox[0])
    ):
        center = (line.bbox[1] + line.bbox[3]) / 2
        height = line.bbox[3] - line.bbox[1]
        matching = []
        for index, row in enumerate(rows):
            if row[0].page != line.page:
                continue
            row_center = median((r.bbox[1] + r.bbox[3]) / 2 for r in row)
            row_height = median(r.bbox[3] - r.bbox[1] for r in row)
            distance = abs(center - row_center)
            if distance <= 0.45 * min(height, row_height):
                matching.append((distance, index))
        if matching:
            rows[min(matching)[1]].append(line)
        else:
            rows.append([line])
    result = []
    for row in rows:
        row.sort(key=lambda x: x.bbox[0])
        first = row[0]
        result.append(
            Line(
                " ".join(r.text for r in row),
                (
                    min(r.bbox[0] for r in row),
                    min(r.bbox[1] for r in row),
                    max(r.bbox[2] for r in row),
                    max(r.bbox[3] for r in row),
                ),
                first.width,
                first.height,
                first.page,
                first.confidence,
                first.source,
            )
        )
    return sorted(result, key=lambda x: (x.page, x.bbox[1], x.bbox[0]))
