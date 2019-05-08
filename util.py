def clean(s):
    s = s.split()
    # remove everything from "-lrb-" to "-rrb-"
    s2 = []
    in_paren = False
    for ixw, w in enumerate(s):
        if w == "-lrb-":
            in_paren = True
        elif w == "-rrb-":
            in_paren = False
        elif w == "-lsb-" or w == "-rsb-":
            continue
        elif len(s2) > 0 and len(w) > 1 and w[0] == '\'':
            s2[-1] = s2[-1] + w
        elif not in_paren and not (w == '<t>' or w == '</t>'):
            s2.append(w)
    return ' '.join(s2)

# `document` and `summary` are arrays
def copy_annotations(document, summary):
    document_annotations = [0] * len(document)
    summary_annotations = [0] * len(summary)
    for summary_index in range(len(summary)):
        best_copy_length_index = -1
        best_copy_length_left = -1
        best_copy_length_right = -1
        for document_index in range(len(document)):
            if summary[summary_index] != document[document_index]:
                continue
            left = right = 0
            while summary_index > left and document_index > left:
                if summary[summary_index - left - 1] == document[document_index - left - 1]:
                    left += 1
                else:
                    break
            while summary_index + right + 1 < len(summary) and document_index + right + 1 < len(document):
                if summary[summary_index + right + 1] == document[document_index + right + 1]:
                    right += 1
                else:
                    break
            copy_length = left + right + 1
            if copy_length > summary_annotations[summary_index]:
                summary_annotations[summary_index] = copy_length
                best_copy_length_index = document_index
                best_copy_length_left = left
                best_copy_length_right = right
        if best_copy_length_index != -1:
            for document_index in range(best_copy_length_index - best_copy_length_left, best_copy_length_index + best_copy_length_right + 1):
                document_annotations[document_index] += 1
    return document_annotations, summary_annotations

def average_copy_length(document, summary):
    if isinstance(document, str):
        document = document.split()
    if isinstance(summary, str):
        summary = summary.split()
    document_annotations, summary_annotations = copy_annotations(document, summary)
    return sum(summary_annotations) / len(summary)
