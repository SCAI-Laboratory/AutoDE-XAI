"""Reusable experimental configurations for new research ideas."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

DE_ONLY_ROLE_PROMPT = """
You are an excellent Python programmer and optimizer researcher. For this
experiment you design only Differential Evolution (DE) algorithms.
"""

DE_ONLY_TASK_PROMPT = """
You are a Python developer working on a new Differential Evolution (DE)
optimizer for continuous black-box minimization.

Hard algorithm constraint:
- generate only a DE algorithm or a direct DE variant
- a valid solution must maintain a population of vectors
- it must create mutant vectors from scaled differences between population
  members, such as rand/1, best/1, current-to-best/1, or pbest/1
- it must create trial vectors with binomial or exponential crossover
- it must select or replace targets with trial vectors using DE-style
  greedy or probabilistic selection

Forbidden algorithm drift:
- do not generate PSO, CMA-ES, genetic algorithms, simulated annealing,
  Bayesian optimization, random-search-only methods, or standalone local
  search methods
- do not use gradients, finite-difference gradient estimates, Nelder-Mead,
  hill climbing, coordinate search, or scipy
- do not import anything except `import numpy as np`

The optimizer is evaluated on the Many Affine BBOB noiseless benchmark. Each
function has bounds from -5.0 to 5.0 and the dimensionality can vary.

The code must contain `__init__(self, budget, dim)` with optional additional
arguments and `def __call__(self, func)`. The optimizer must use at most
`self.budget` calls to `func`.

Budget-safety requirements:
- maintain an explicit `eval_count`
- increment it immediately after every `func(...)` call
- check `eval_count < self.budget` before every objective evaluation
- set `pop_size = min(pop_size, self.budget)` and handle tiny budgets
- force at least one crossover dimension for every trial vector
- clip every trial vector to the objective bounds before evaluation
"""

DE_ONLY_EXAMPLE_PROMPT = """
An example of valid budget-safe DE code is:
```python
import numpy as np


class BudgetSafeDE:
    def __init__(self, budget=10000, dim=10, pop_size=24, F=0.7, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0
        pop_size = min(max(1, self.pop_size), self.budget)

        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if eval_count >= self.budget:
                break
            fitness[i] = func(population[i])
            eval_count += 1

        best_idx = int(np.argmin(fitness))
        self.f_opt = float(fitness[best_idx])
        self.x_opt = population[best_idx].copy()

        if pop_size < 4:
            return self.f_opt, self.x_opt

        while eval_count < self.budget:
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                choices = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(choices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])

                mask = np.random.random(self.dim) < self.CR
                mask[np.random.randint(self.dim)] = True
                trial = np.where(mask, mutant, population[i])
                trial = np.clip(trial, lb, ub)

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness <= fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = float(trial_fitness)
                        self.x_opt = trial.copy()

        return self.f_opt, self.x_opt
```
"""

DE_ONLY_FORMAT_PROMPT = """
Give an excellent and novel Differential Evolution (DE) variant and a one-line
description of its DE idea. Give the response in the format:
# Description: <short-description>
# Code:
```python
<code>
```
"""

DE_ONLY_PROMPT_KWARGS = {
    "role_prompt": DE_ONLY_ROLE_PROMPT,
    "task_prompt": DE_ONLY_TASK_PROMPT,
    "example_prompt": DE_ONLY_EXAMPLE_PROMPT,
    "output_format_prompt": DE_ONLY_FORMAT_PROMPT,
}

SAGE_QD_DE_MUTATION_PROMPTS = [
    """
Refine the selected solution while keeping it a Differential Evolution (DE)
algorithm.

Required DE skeleton:
- maintain a population of vectors
- create mutant vectors from differences between population members
- apply binomial or exponential crossover
- select between target and trial vectors

Research pressure:
- preserve the parent only if it is already a valid DE variant
- if the parent is not valid DE, rewrite it into a valid DE variant
- improve one weak mechanism only
- make F and CR react to success, diversity, or stagnation
- keep all bounds handling explicit and budget-safe
- do not add local search, gradients, scipy, or a non-DE optimizer
""",
    """
Generate a new Differential Evolution (DE) variant that is deliberately
different from the archive so far.

Required DE skeleton:
- population initialization inside the objective bounds
- mutation based on vector differences
- crossover into a trial vector
- greedy or probabilistic replacement

Research pressure:
- combine two mutation strategies, such as rand/1 and current-to-best/1
- add restart, opposition, archive, or crowding behavior only if useful
- avoid large external dependencies
- never exceed the function evaluation budget
- do not generate PSO, CMA-ES, GA, simulated annealing, Bayesian
  optimization, or standalone local search
""",
    """
Create a compact DE variant that targets robustness on many BBOB landscapes.

Required DE skeleton:
- population, mutation, crossover, selection
- adaptive F and CR
- clear stagnation recovery

Research pressure:
- balance exploration and exploitation inside the DE loop
- include a simple success memory or rank-based parent choice
- keep the code readable enough for future mutations
- do not use gradients, scipy, Nelder-Mead, hill climbing, or coordinate search
""",
]

_DE_IMPLEMENTATION_GUARDRAILS = """

Implementation guardrails:
- use only `import numpy as np`; do not import scipy or other packages
- initialize with `pop_size = min(pop_size, budget)` and handle tiny budgets
- maintain an `eval_count`; increment it after every `func(...)` call
- check `eval_count < self.budget` before every objective evaluation
- opposition, archive updates, and restarts must also be budget-safe, or
  should be skipped when budget is low
- do not add helper local-search, gradient, or scipy-based routines
- always force at least one crossover dimension and clip every trial to bounds
- if feedback mentions `numpy._core.numeric`, treat it as infrastructure
  feedback, not as a reason to add dependencies or abandon DE
"""

SAGE_QD_DE_MUTATION_PROMPTS = [
    prompt + _DE_IMPLEMENTATION_GUARDRAILS
    for prompt in SAGE_QD_DE_MUTATION_PROMPTS
]

_ADAPTATION_TERMS = (
    "adaptive",
    "adapt",
    "success",
    "memory",
    "history",
    "shade",
    "jde",
    "pbest",
    "rank",
    "tau",
)

_RECOVERY_TERMS = (
    "restart",
    "stagnation",
    "stagnant",
    "diversity",
    "opposition",
    "archive",
    "crowding",
    "reinitialize",
    "cauchy",
    "levy",
)

_DE_STRATEGY_TERMS = (
    "rand",
    "best",
    "current_to_best",
    "current-to-best",
    "pbest",
    "binomial",
    "exponential",
    "j_rand",
    "archive",
    "opposition",
    "success_memory",
    "shade",
    "jde",
    "rank",
)

_DE_STRUCTURE_TERMS = (
    "population",
    "mutation",
    "crossover",
    "selection",
    "trial",
    "mutant",
    "bounds",
    "clip",
)

_FORBIDDEN_DRIFT_TERMS = (
    "scipy",
    "gradient",
    "finite_difference",
    "nelder",
    "hill",
    "coordinate",
    "particle",
    "swarm",
    "cma",
    "anneal",
    "bayes",
    "genetic",
    "random_search",
)


def _solution_metadata(solution: Any, key: str) -> Any:
    getter = getattr(solution, "get_metadata", None)
    if getter is None:
        return None
    try:
        return getter(key)
    except Exception:
        return None


def _add_solution_metadata(solution: Any, key: str, value: Any) -> None:
    setter = getattr(solution, "add_metadata", None)
    if setter is None:
        return
    try:
        setter(key, value)
    except Exception:
        return


def _extract_features(solution: Any, code: str) -> Mapping[str, float]:
    features = _solution_metadata(solution, "ast_features")
    if isinstance(features, Mapping):
        return features

    try:
        from llamea.ast_features import extract_ast_features

        features = extract_ast_features(code)
    except Exception:
        return {}

    _add_solution_metadata(solution, "ast_features", dict(features))
    return features


def _feature_value(features: Mapping[str, float], *names: str) -> float:
    for name in names:
        value = features.get(name)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            return value_float
    return 0.0


def _keyword_score(code: str, terms: tuple[str, ...]) -> float:
    code_lower = code.lower()
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", code_lower)
    token_counts = {token: tokens.count(token) for token in set(tokens)}

    count = 0
    for term in terms:
        term_lower = term.lower()
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", term_lower):
            count += token_counts.get(term_lower, 0)
        else:
            count += code_lower.count(term_lower)
    return math.log1p(float(count))


def de_quality_diversity_descriptor(solution: Any) -> tuple[float, float, float, float]:
    """Return a MAP-Elites descriptor for generated DE-style algorithms.

    The descriptor intentionally mixes light semantic DE signals with AST
    structure. This gives SAGE-QD-DE niches for adaptive-control heavy,
    recovery/diversity heavy, DE-strategy diverse, and structurally complex
    variants without encouraging non-DE local-search hybrids.
    """

    code = getattr(solution, "code", "") or ""
    features = _extract_features(solution, code)

    adaptation = _keyword_score(code, _ADAPTATION_TERMS)
    recovery = _keyword_score(code, _RECOVERY_TERMS)
    strategy_diversity = _keyword_score(code, _DE_STRATEGY_TERMS)
    de_structure = _keyword_score(code, _DE_STRUCTURE_TERMS)
    drift_penalty = _keyword_score(code, _FORBIDDEN_DRIFT_TERMS)

    total_complexity = _feature_value(features, "total_complexity")
    mean_depth = _feature_value(features, "Mean Depth", "mean_depth")
    entropy = _feature_value(features, "Degree Entropy", "degree_entropy")
    token_count = _feature_value(features, "total_token_count")

    structure = math.log1p(
        max(0.0, total_complexity)
        + max(0.0, mean_depth)
        + max(0.0, entropy)
        + 0.01 * max(0.0, token_count)
        + de_structure
    )

    return (
        max(0.0, adaptation - drift_penalty),
        max(0.0, recovery - drift_penalty),
        max(0.0, strategy_diversity - drift_penalty),
        max(0.0, structure - drift_penalty),
    )


def make_sage_qd_de_method(
    llm: Any,
    budget: int,
    *,
    name: str = "SAGE-QD-DE",
    n_parents: int = 6,
    n_offspring: int = 18,
    map_elites_bins: tuple[int, int, int, int] = (4, 4, 4, 4),
    tournament_size: int = 3,
    **kwargs: Any,
):
    """Build the SAGE-QD-DE LLaMEA method.

    SAGE-QD-DE combines SAGE feature-guided mutation with MAP-Elites
    niching. The hypothesis is that feature guidance improves local mutation
    direction while the quality-diversity archive keeps structurally distinct
    DE families alive long enough to mature.
    """

    from .methods import LLaMEA

    config = {
        **DE_ONLY_PROMPT_KWARGS,
        "mutation_prompts": list(SAGE_QD_DE_MUTATION_PROMPTS),
        "n_parents": n_parents,
        "n_offspring": n_offspring,
        "elitism": True,
        "feature_guided_mutation": True,
        "niching": "map_elites",
        "behavior_descriptor": de_quality_diversity_descriptor,
        "map_elites_bins": map_elites_bins,
        "parent_selection": "tournament",
        "tournament_size": tournament_size,
    }
    config.update(kwargs)

    return LLaMEA(llm, budget=budget, name=name, **config)


__all__ = [
    "DE_ONLY_EXAMPLE_PROMPT",
    "DE_ONLY_FORMAT_PROMPT",
    "DE_ONLY_PROMPT_KWARGS",
    "DE_ONLY_ROLE_PROMPT",
    "DE_ONLY_TASK_PROMPT",
    "SAGE_QD_DE_MUTATION_PROMPTS",
    "de_quality_diversity_descriptor",
    "make_sage_qd_de_method",
]
