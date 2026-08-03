import math

from iohblade.llm import Dummy_LLM
from iohblade.research_ideas import (
    DE_ONLY_TASK_PROMPT,
    de_quality_diversity_descriptor,
    make_sage_qd_de_method,
)
from iohblade.solution import Solution

DE_CODE = """
import numpy as np


class AdaptiveDE:
    def __init__(self, budget=1000, dim=5):
        self.budget = budget
        self.dim = dim
        self.memory = []

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (10, self.dim))
        scores = np.array([func(x) for x in pop])
        best = pop[np.argmin(scores)]
        f_value = 0.5
        cr = 0.9
        for _ in range(max(0, self.budget - len(pop))):
            ids = np.random.choice(len(pop), 3, replace=False)
            mutant = pop[ids[0]] + f_value * (pop[ids[1]] - pop[ids[2]])
            trial = np.where(np.random.rand(self.dim) < cr, mutant, best)
            trial = np.clip(trial, func.bounds.lb, func.bounds.ub)
            score = func(trial)
            worst = np.argmax(scores)
            if score < scores[worst]:
                pop[worst] = trial
                scores[worst] = score
                self.memory.append((f_value, cr))
                best = pop[np.argmin(scores)]
        return float(np.min(scores)), best
"""


def test_de_quality_diversity_descriptor_is_finite():
    solution = Solution(code=DE_CODE, name="AdaptiveDE")

    descriptor = de_quality_diversity_descriptor(solution)

    assert len(descriptor) == 4
    assert all(math.isfinite(value) for value in descriptor)
    assert solution.get_metadata("ast_features") is not None


def test_descriptor_reflects_recovery_logic():
    plain = Solution(code=DE_CODE)
    recovery = Solution(
        code=DE_CODE
        + """
def restart_on_stagnation(population, diversity):
    if diversity < 1e-6:
        return "restart with opposition archive"
    return "keep population"
"""
    )

    plain_descriptor = de_quality_diversity_descriptor(plain)
    recovery_descriptor = de_quality_diversity_descriptor(recovery)

    assert recovery_descriptor[1] > plain_descriptor[1]


def test_descriptor_prefers_de_strategy_over_local_search_drift():
    de_strategy = Solution(
        code=DE_CODE
        + """
def use_pbest_archive(population, archive, success_memory):
    return "pbest rand current-to-best binomial opposition"
"""
    )
    local_search_drift = Solution(
        code=DE_CODE
        + """
def gradient_polish_with_scipy(x):
    return "gradient scipy nelder hill coordinate"
"""
    )

    de_descriptor = de_quality_diversity_descriptor(de_strategy)
    drift_descriptor = de_quality_diversity_descriptor(local_search_drift)

    assert de_descriptor[2] > drift_descriptor[2]


def test_make_sage_qd_de_method_sets_expected_kwargs():
    method = make_sage_qd_de_method(Dummy_LLM(), budget=24)

    assert method.name == "SAGE-QD-DE"
    assert method.budget == 24
    assert method.task_prompt == DE_ONLY_TASK_PROMPT
    assert method.kwargs["feature_guided_mutation"] is True
    assert method.kwargs["niching"] == "map_elites"
    assert method.kwargs["parent_selection"] == "tournament"
    assert method.kwargs["behavior_descriptor"] is de_quality_diversity_descriptor
