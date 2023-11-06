import json
import os

import numpy as np

from src.utils.util import ROOT_DIR
from src.constants import NATOPS_TEXT, NATOPS_TO_TEXT_MAP, NATOPS_WO_INDEP


class TemplateFormatter(object):
    def __init__(
        self,
        setting_nr,
        neg_token,
        num_templates=5,
        num_questions=5,
        template_id=0,
        question_id=0,
        randomize=True,
    ):
        self.question_types = NATOPS_TEXT
        self.op_to_type_map = NATOPS_TO_TEXT_MAP  # what about #
        self.neg_token = neg_token
        self.data_path = os.path.join(ROOT_DIR, "data", "handwritten_questions", "setting" + str(setting_nr))

        self.randomize = randomize
        self.template_id = template_id
        self.question_id = question_id

        self.questions = {
            q_type: self._read_questions_for_operator(q_type, num_questions) for q_type in self.question_types
        }
        print(self.questions)

    def _read_questions_for_operator(self, operator, num_questions):
        questions = []
        file = os.path.join(self.data_path, operator + ".csv")
        with open(file, "r") as f_in:
            lines = f_in.readlines()
            for line in lines:
                questions.append(line.strip())
        if self.randomize:
            return [np.random.choice(questions)]
        elif self.question_id >= 0:
            return [questions[self.question_id]]
        else:
            return questions[:num_questions]

    def _apply_component_to_templates(self, templates, component, component_name):
        applied_templates = []
        for template in templates:
            for question in component:
                applied_templates.append(template.replace("{{" + component_name + "}}", question))

        return applied_templates

    def apply_templates_to_sample(self, claim, claim_span, evidence, ev_span, operator):
        claim = [claim]
        evidence = [evidence]
        claim_span = [claim_span]
        applied_templates = self._apply_component_to_templates(self.templates, claim, "claim")
        applied_templates = self._apply_component_to_templates(applied_templates, evidence, "evidence")
        applied_templates = self._apply_component_to_templates(
            applied_templates, self.questions[self.op_to_type_map[operator]], "question"
        )

        # Span must be inserted after question since the question specifies the claim span position
        applied_templates = self._apply_component_to_templates(applied_templates, claim_span, "span")

        answer_list = [ev_span] * len(applied_templates)

        return [applied_templates, answer_list]

    def apply_templates_to_sample_all_ops(self, claim_span, ev_span, operator, claim, evidence):
        claim_span = [claim_span]
        ev_span = [ev_span]
        claim = [claim.strip()]
        evidence = [evidence.strip()]
        ops = NATOPS_WO_INDEP

        applied_templates_collection = []
        answer_list = []

        for op in ops:
            applied_templates_op = self._apply_component_to_templates(
                    self.questions[self.op_to_type_map[op]], claim_span, "span"
                )
            applied_templates_op = self._apply_component_to_templates(applied_templates_op, ev_span, "evidence")
            applied_templates_collection.append(applied_templates_op)

        for i in range(len(NATOPS_WO_INDEP)):
            if NATOPS_WO_INDEP[i] == operator:
                answer_list += ["Yes"] * len(applied_templates_collection[i])
            else:
                answer_list += ["No"] * len(applied_templates_collection[i])

        applied_templates = [item for sublist in applied_templates_collection for item in sublist]

        return [applied_templates, answer_list]
