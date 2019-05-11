import spacy

def clean(s):
    s = s.split()
    # remove everything from "-lrb-" to "-rrb-"
    s2 = []
    in_paren = False
    for ixw, w in enumerate(s):
        if(w=="-lrb-"):
            in_paren=True
        elif(w=='-rrb-'):
            in_paren=False
        elif(w=="-lsb-" or w=="-rsb-"):
            continue
        elif(ixw <= len(s)-2 and w=="new" and s[ixw+1]==":"):
            continue
        elif(ixw <= len(s)-1 and w==":" and s[ixw-1]=="new"):
            continue
        elif(len(w) > 1 and w[0] == '\''):
            if(w[-1] == '\''):
                s2.append('\'')
                s2.append(w[1:-1])
                s2.append('\'')
            elif len(s2) > 0:
                s2[-1] = s2[-1]+w
            else:
                s2.append(w)
                
        elif not in_paren and not (w == '<t>' or w == '</t>'):
            s2.append(w)
        
    return ' '.join(s2)

def get_conj(noun):
    stack = [noun]
    conj_list = [noun]
    while len(stack) > 0:
        noun = stack.pop()
        for child in noun.children:
            if child.dep_ == "conj":
                conj_list.append(child)
                stack.append(child)
    return conj_list

def get_actors(verb):
    actors = []
    for child in verb.children:
        if child.dep_ == "nsubj":
            actors.extend(get_conj(child))
        elif child.dep_ == "agent":
            # passive, look for true actor
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    actors.extend(get_conj(child))
    if verb.dep_ == "acl":
        if verb.text[-3:] == "ing":
            actors.append(verb.head)
    return actors

def get_acteds(verb):
    acteds = []
    for child in verb.children:
        if child.dep_ == "dobj" or child.dep_ == "nsubjpass":
            acteds.extend(get_conj(child))
    if verb.dep_ == "acl":
        if verb.text[-3:] != "ing":
            acteds.append(verb.head)
    return acteds


def equivalent(w1, w2):
    if isinstance(w1, spacy.tokens.token.Token):
        w1 = w1.text
    if isinstance(w2, spacy.tokens.token.Token):
        w2 = w2.text
    return w1 == w2


# `document` and `summary` are arrays
def copy_annotations(document, summary):
    document_annotations = [0] * len(document)
    summary_annotations = [0] * len(summary)
    for summary_index in range(len(summary)):
        best_copy_length_index = -1
        best_copy_length_left = -1
        best_copy_length_right = -1
        for document_index in range(len(document)):
            if not equivalent(summary[summary_index], document[document_index]):
                continue
            left = right = 0
            while summary_index > left and document_index > left:
                if equivalent(summary[summary_index - left - 1], document[document_index - left - 1]):
                    left += 1
                else:
                    break
            while summary_index + right + 1 < len(summary) and document_index + right + 1 < len(document):
                if equivalent(summary[summary_index + right + 1], document[document_index + right + 1]):
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

def build_minimal_sentence(relation):
    verb = relation[0]
    actors = relation[1]
    acteds = relation[2]
    return " ".join([token.text for token in actors + [verb] + acteds])

