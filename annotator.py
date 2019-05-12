import util
from knowledge_graph import KnowledgeGraph
from termcolor import colored

class Annotator:
    copied_word = 1
    missing_dependencies = 2 << KnowledgeGraph.missing_dependencies
    missing_verb = 2 << KnowledgeGraph.missing_verb
    entailment = 2 << KnowledgeGraph.entailment
    entailment_bert = 2 << KnowledgeGraph.entailment_bert
    contradiction = 2 << KnowledgeGraph.contradiction
    invalid_simplification = 2 << KnowledgeGraph.invalid_simplification
    contradiction_bert = 2 << KnowledgeGraph.contradiction_bert

    def __init__(self, document, summary):
        self.document = document
        self.summary = summary
        self.document_annotations, self.summary_annotations = util.copy_annotations(self.document, self.summary)
        for i in range(len(self.document_annotations)):
            if self.document_annotations[i]:
                self.document_annotations[i] = Annotator.copied_word
            else:
                self.document_annotations[i] = 0
        for i in range(len(self.summary)):
            if self.summary_annotations[i]:
                self.summary_annotations[i] = Annotator.copied_word
            else:
                self.summary_annotations[i] = 0

    def annotate(self, relation, query_result):
        color = 2 << query_result[0]
        verb = relation[0]
        actors = relation[1]
        acteds = relation[2]
        self.summary_annotations[verb.i] |= color
        for actor in actors:
            self.summary_annotations[actor.i] |= color
        for acted in acteds:
            self.summary_annotations[acted.i] |= color
        for proof in query_result[1]:
            relation = proof[0]
            print(relation)
            verb = relation[0]
            actors = relation[1]
            acteds = relation[2]
            self.document_annotations[verb.i] |= color
            for actor in actors:
                self.document_annotations[actor.i] |= color
            for acted in acteds:
                self.document_annotations[acted.i] |= color

    def get_color(self, annotation):
        if annotation & Annotator.contradiction:
            return 'red'
        if annotation & Annotator.contradiction_bert:
            return 'red'
        if annotation & Annotator.invalid_simplification:
            return 'magenta'
        if annotation & Annotator.missing_dependencies:
            return 'yellow'
        if annotation & Annotator.missing_verb:
            return 'yellow'
        if annotation & Annotator.entailment:
            return 'blue'
        if annotation & Annotator.entailment_bert:
            return 'blue'
        return None
    
    def get_highlight(self, annotation):
        if annotation & Annotator.copied_word:
            return 'on_grey'
        return None

    def annotated(self):
        annotated_document = list()
        for word, annotation in zip(self.document, self.document_annotations):
            word = word.text
            annotated_document.append(colored(word, self.get_color(annotation),
                self.get_highlight(annotation)))
        annotated_summary = list()
        for word, annotation in zip(self.summary, self.summary_annotations):
            word = word.text
            annotated_summary.append(colored(word, self.get_color(annotation),
                self.get_highlight(annotation)))
        return annotated_document, annotated_summary
