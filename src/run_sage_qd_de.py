# import os

# from iohblade.experiment import MA_BBOB_Experiment
# from iohblade.llm import Ollama_LLM
# from iohblade.loggers import ExperimentLogger
# from iohblade.methods import LLaMEA
# from iohblade.research_ideas import (
#     DE_ONLY_PROMPT_KWARGS,
#     SAGE_QD_DE_MUTATION_PROMPTS,
#     make_sage_qd_de_method,
# )


# def _env_flag(name: str, default: bool) -> bool:
#     value = os.getenv(name)
#     if value is None:
#         return default
#     return value.lower() in {"1", "true", "yes", "on"}


# def build_methods(llm, budget: int):
#     common = {
#         **DE_ONLY_PROMPT_KWARGS,
#         "mutation_prompts": list(SAGE_QD_DE_MUTATION_PROMPTS),
#         "n_parents": 6,
#         "n_offspring": 18,
#         "elitism": True,
#     }

#     baseline_de = LLaMEA(
#         llm,
#         budget=budget,
#         name="DE-LLaMEA",
#         **common,
#     )
#     sage_de = LLaMEA(
#         llm,
#         budget=budget,
#         name="SAGE-DE",
#         feature_guided_mutation=True,
#         parent_selection="tournament",
#         tournament_size=3,
#         **common,
#     )
#     sage_qd_de = make_sage_qd_de_method(
#         llm,
#         budget=budget,
#         n_parents=common["n_parents"],
#         n_offspring=common["n_offspring"],
#     )

#     return [baseline_de, sage_de, sage_qd_de]


# def main():
#     debug = _env_flag("SAGE_QD_DEBUG", True)
#     default_budget = "24" if debug else "204"
#     budget = int(os.getenv("SAGE_QD_BUDGET", default_budget))

#     llm = Ollama_LLM(model=os.getenv("OLLAMA_MODEL", "qwen3-coder:30b"))
#     methods = build_methods(llm, budget)

#     if debug:
#         logger_dir = os.getenv("SAGE_QD_RESULTS", "results/MABBOB_sage_qd_de_debug")
#         experiment = MA_BBOB_Experiment(
#             methods=methods,
#             training_instances=[0, 1],
#             runs=1,
#             seeds=[1],
#             dims=[2],
#             budget_factor=200,
#             budget=budget,
#             eval_timeout=60,
#             show_stdout=True,
#             log_stdout=False,
#             exp_logger=ExperimentLogger(logger_dir),
#             n_jobs=int(os.getenv("SAGE_QD_N_JOBS", "2")),
#         )
#     else:
#         logger_dir = os.getenv("SAGE_QD_RESULTS", "results/MABBOB_sage_qd_de")
#         experiment = MA_BBOB_Experiment(
#             methods=methods,
#             training_instances=range(0, 10),
#             runs=5,
#             seeds=[1, 2, 3, 4, 5],
#             dims=[10],
#             budget_factor=5000,
#             budget=budget,
#             eval_timeout=600,
#             show_stdout=False,
#             log_stdout=True,
#             exp_logger=ExperimentLogger(logger_dir),
#             n_jobs=int(os.getenv("SAGE_QD_N_JOBS", "5")),
            
#         )
        

#     experiment()
    


# if __name__ == "__main__":
#     main()

##############################
import os
import sys

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Ollama_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA
from iohblade.research_ideas import (
    DE_ONLY_PROMPT_KWARGS,
    SAGE_QD_DE_MUTATION_PROMPTS,
    make_sage_qd_de_method,
)


def _env_flag(name: str, default: bool) -> bool:
    """Parse boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse integer environment variable with error handling."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Warning: Invalid value for {name}='{value}', using default {default}")
        return default


def build_methods(llm, budget: int):
    """Build optimization methods for the experiment."""
    common = {
        **DE_ONLY_PROMPT_KWARGS,
        "mutation_prompts": list(SAGE_QD_DE_MUTATION_PROMPTS),
        "n_parents": 6,
        "n_offspring": 18,
        "elitism": True,
    }

    baseline_de = LLaMEA(
        llm,
        budget=budget,
        name="DE-LLaMEA",
        **common,
    )
    sage_de = LLaMEA(
        llm,
        budget=budget,
        name="SAGE-DE",
        feature_guided_mutation=True,
        parent_selection="tournament",
        tournament_size=3,
        **common,
    )
    sage_qd_de = make_sage_qd_de_method(
        llm,
        budget=budget,
        n_parents=common["n_parents"],
        n_offspring=common["n_offspring"],
    )

    return [baseline_de, sage_de, sage_qd_de]


def main():
    """Main execution function."""
    try:
        debug = _env_flag("SAGE_QD_DEBUG", True)
        default_budget = "24" if debug else "204"
        budget = _env_int("SAGE_QD_BUDGET", int(default_budget))

        llm = Ollama_LLM(model=os.getenv("OLLAMA_MODEL", "qwen3-coder:30b"))
        methods = build_methods(llm, budget)

        if debug:
            logger_dir = os.getenv("SAGE_QD_RESULTS", "results/MABBOB_sage_qd_de_debug")
            experiment = MA_BBOB_Experiment(
                methods=methods,
                training_instances=[0, 1],
                runs=1,
                seeds=[1],
                dims=[2],
                budget_factor=200,
                budget=budget,
                eval_timeout=60,
                show_stdout=True,
                log_stdout=False,
                exp_logger=ExperimentLogger(logger_dir),
                n_jobs=_env_int("SAGE_QD_N_JOBS", 2)
            )
        else:
            logger_dir = os.getenv("SAGE_QD_RESULTS", "results/MABBOB_sage_qd_de")
            experiment = MA_BBOB_Experiment(
                methods=methods,
                training_instances=range(0, 10),
                runs=5,
                seeds=[1, 2, 3, 4, 5],
                dims=[10],
                budget_factor=5000,
                budget=budget,
                eval_timeout=600,
                show_stdout=False,
                log_stdout=True,
                exp_logger=ExperimentLogger(logger_dir),
                n_jobs=_env_int("SAGE_QD_N_JOBS", 5)
            )

        experiment()
        
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()