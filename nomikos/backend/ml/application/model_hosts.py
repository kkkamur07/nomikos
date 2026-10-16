"""Which **execution target**s a model permits.

**Host eligibility** lives in the **Registry** (``registry.yaml``), keyed by
**registry model id**. This module is the one place that turns it into the set of
targets a job may choose. It *constrains*; it never chooses - that is
:func:`backend.ml.domain.execution.choose_execution_target`.

The mapping is deliberately asymmetric, and follows the glossary wording:

* ``remote`` - "only on a hosted server". Not a **lite model tier**, so ``local``
  is out.
* ``local`` - "may run on the researcher's machine". A **lite model tier** is
  sized for a laptop CPU; a hosted worker runs the same package on a bigger
  machine, so it can run one too. Both targets stay open.
* ``any`` - either.

An id the Registry does not know is treated as ``remote``. We cannot show it is
laptop-sized, and "not demonstrably lite" is the honest reading of the rule -
the alternative silently ships an unknown model to a researcher's CPU.
"""

from __future__ import annotations

from backend.core.settings.ml import get_inference_settings
from backend.ml.domain.execution import ALL_EXECUTION_TARGETS, ExecutionTarget
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from nomikos_inference.contracts.common import HostEligibility
from nomikos_inference.registry import load_registry

# The registry model id used when a job carries no catalog model. Kept here
# rather than imported from the dispatcher: this is a property of the Registry,
# and the dispatcher is being rewritten around it.
#
# There are four transcribe models now (Greek, Armenian, Syriac, Coptic) and no reason
# to think a job carrying no catalog model wants the Syriac one. The entry stays
# unchanged anyway, because what this module asks of it is narrower than it
# looks: the id is handed to `host_eligibility_for`, and all four Calamari
# entries are `local`, so every candidate answers that question identically.
# Naming a different script here would change no behaviour and would still be a
# guess.
#
# The guess that does matter, which weights transcribe a page whose job named no
# model, is not fixable by editing this line: any single default is wrong for three
# of the four scripts. That belongs upstream, in making the caller carry a
# catalog model. Dropping the key would be the quiet change, not the honest one:
# `registry_model_id_for` would return None, unknown reads as `remote` by design,
# and model-less transcribe jobs would stop being eligible for a researcher's
# laptop without anything saying so.
DEFAULT_REGISTRY_MODEL_IDS: dict[InferenceTask, str] = {
    InferenceTask.segment: "blla-segment",
    InferenceTask.transcribe: "syriac-calamari-v2",
}

_REGISTRY_SCHEME = "registry://"

_ELIGIBLE_TARGETS: dict[HostEligibility, frozenset[ExecutionTarget]] = {
    HostEligibility.local: ALL_EXECUTION_TARGETS,
    HostEligibility.any: ALL_EXECUTION_TARGETS,
    HostEligibility.remote: frozenset({ExecutionTarget.cloud}),
}


def registry_model_id_for(model: InferenceModel | None, *, task: InferenceTask) -> str | None:
    """The **registry model id** a job will run, or ``None`` if it cannot be told.

    A catalog model points at the Registry through ``artifact_ref``
    (``registry://<registry_model_id>?tag=<tag>``). Anything else - a legacy ref,
    a provider-specific path - is not a registry model id and is reported as
    unknown rather than guessed at.
    """
    if model is None:
        return DEFAULT_REGISTRY_MODEL_IDS.get(task)
    ref = (model.artifact_ref or "").strip()
    if not ref.startswith(_REGISTRY_SCHEME):
        return None
    remainder = ref[len(_REGISTRY_SCHEME) :]
    registry_model_id = remainder.split("?", 1)[0].split("/", 1)[0]
    return registry_model_id or None


def host_eligibility_for(registry_model_id: str | None) -> HostEligibility:
    if registry_model_id is None:
        return HostEligibility.remote
    try:
        registry = load_registry(get_inference_settings().inference_registry_path)
    except (OSError, ValueError):
        # An unreadable or invalid Registry must not silently widen eligibility.
        return HostEligibility.remote
    entry = registry.models.get(registry_model_id)
    if entry is None:
        return HostEligibility.remote
    return entry.host_eligibility


def eligible_targets_for_model(
    model: InferenceModel | None, *, task: InferenceTask
) -> frozenset[ExecutionTarget]:
    """The **execution target**s this model may be run on."""
    return _ELIGIBLE_TARGETS[host_eligibility_for(registry_model_id_for(model, task=task))]
