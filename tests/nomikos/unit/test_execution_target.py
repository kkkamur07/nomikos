"""Choosing one **execution target** from preference, eligibility, and **capacity**.

Everything here runs the real code on real values - there is nothing to fake,
because the decision was deliberately built as a pure function of three inputs.
Where those inputs come from (device rows, the Registry, the account setting) is
exercised against live Postgres in
``tests/nomikos/integration/test_execution_target.py``.

What is pinned:

* **preference first, then the other host, and nothing else.** No scoring, no
  queue depth. A second host is tried only because the first cannot take work.
* **substitution is recorded, not swallowed.** The decision carries the host the
  researcher asked for as well as the one they got, because announcing the
  downgrade *is* the feature (ADR 0002).
* **refusal has a reason.** Both refusals name their situation: "nothing is
  running" and "nothing this model can run on is running" are different problems
  with different fixes.
* **eligibility constrains, it does not choose.** A model that is not a **lite
  model tier** cannot be sent to a laptop even when the laptop is the preference
  and the laptop is available.

Also asserted: ``local_only`` is not a member of the enum. It is the one thing a
future reader is most likely to add back, and the reason it is gone is a decision
(ADR 0002), not an oversight.
"""

from __future__ import annotations

import pytest

from backend.core.exceptions import ConflictError
from backend.ml.application.model_hosts import (
    DEFAULT_REGISTRY_MODEL_IDS,
    eligible_targets_for_model,
    host_eligibility_for,
    registry_model_id_for,
)
from backend.ml.domain.execution import (
    NO_CAPACITY_MESSAGE,
    ExecutionRequest,
    ExecutionTarget,
    choose_execution_target,
)
from backend.ml.infrastructure.orm_models import InferenceModel, InferenceTask
from nomikos_inference.contracts.common import HostEligibility

LOCAL = ExecutionTarget.local
CLOUD = ExecutionTarget.cloud
EITHER = frozenset({LOCAL, CLOUD})


def _request(preferred: ExecutionTarget, *available: ExecutionTarget) -> ExecutionRequest:
    return ExecutionRequest(preferred=preferred, available=frozenset(available))


# --- The enum ---


def test_the_only_execution_targets_are_local_and_cloud() -> None:
    """``local_only`` was retired by ADR 0002 and must not grow back.

    Its headline justification - manuscripts never leave the machine - was never
    true, since page images already live in the platform's media store; and it
    was the one mode that could leave a job with no terminal outcome.
    """
    assert {target.value for target in ExecutionTarget} == {"local", "cloud"}


# --- Preference is honoured when the preferred host can take the work ---


@pytest.mark.parametrize("preferred", [LOCAL, CLOUD])
def test_the_preferred_host_wins_when_it_has_capacity(preferred: ExecutionTarget) -> None:
    decision = choose_execution_target(_request(preferred, LOCAL, CLOUD), eligible=EITHER)

    assert decision.target is preferred
    assert decision.preferred is preferred


# --- Substitution: the other host, and the job says so ---


@pytest.mark.parametrize(
    ("preferred", "available", "expected"),
    [
        (LOCAL, CLOUD, CLOUD),
        (CLOUD, LOCAL, LOCAL),
    ],
)
def test_an_unavailable_preference_is_substituted_and_recorded(
    preferred: ExecutionTarget, available: ExecutionTarget, expected: ExecutionTarget
) -> None:
    """Never silently. The preferred host stays on the decision precisely so the
    researcher can be told the job did not go where they asked."""
    decision = choose_execution_target(_request(preferred, available), eligible=EITHER)

    assert decision.target is expected
    assert decision.preferred is preferred


# --- Refusal: no row is created for a host nobody claims from ---


@pytest.mark.parametrize("preferred", [LOCAL, CLOUD])
def test_no_capacity_anywhere_is_refused_with_a_reason(preferred: ExecutionTarget) -> None:
    with pytest.raises(ConflictError) as refusal:
        choose_execution_target(_request(preferred), eligible=EITHER)

    assert str(refusal.value) == NO_CAPACITY_MESSAGE
    # The message has to name the situation, not just fail: "start the agent" is
    # the only action available to the researcher, and nothing else will say it.
    assert "no cloud worker is online" in str(refusal.value)
    assert "nomikos agent" in str(refusal.value)


def test_capacity_on_an_ineligible_host_is_refused_as_a_model_problem() -> None:
    """A different refusal, because it has a different fix.

    Starting the agent will not help: the host that is up is not one this model
    may run on, so the reason names the model rather than the hosts.
    """
    with pytest.raises(ConflictError) as refusal:
        choose_execution_target(_request(LOCAL, LOCAL), eligible=frozenset({CLOUD}))

    assert "selected model" in str(refusal.value)
    assert str(refusal.value) != NO_CAPACITY_MESSAGE


# --- Eligibility constrains; it never chooses ---


def test_a_model_that_is_not_a_lite_tier_is_ineligible_for_local() -> None:
    """Both hosts are up and the researcher prefers their laptop. The model still
    goes to the cloud, because **host eligibility** is a constraint on the choice
    and not an input the preference can outvote."""
    decision = choose_execution_target(_request(LOCAL, LOCAL, CLOUD), eligible=frozenset({CLOUD}))

    assert decision.target is CLOUD


# --- Resolving a model to its registry entry ---


def _model(artifact_ref: str, task: InferenceTask = InferenceTask.transcribe) -> InferenceModel:
    return InferenceModel(
        name="m", provider="p", task=task, artifact_ref=artifact_ref, default_params={}
    )


@pytest.mark.parametrize(
    ("artifact_ref", "expected"),
    [
        ("registry://syriac-calamari-v2?tag=stable", "syriac-calamari-v2"),
        ("registry://greek-calamari-v1?tag=stable", "greek-calamari-v1"),
        ("registry://armenian-calamari-v1?tag=stable", "armenian-calamari-v1"),
        ("registry://coptic-calamari-v1?tag=stable", "coptic-calamari-v1"),
        ("registry://blla-segment", "blla-segment"),
    ],
)
def test_a_registry_ref_resolves_to_its_registry_model_id(artifact_ref: str, expected: str) -> None:
    assert registry_model_id_for(_model(artifact_ref), task=InferenceTask.transcribe) == expected


def test_a_model_that_does_not_point_at_the_registry_cannot_be_shown_to_be_lite() -> None:
    """Not demonstrably a **lite model tier**, so not eligible for a laptop.

    The alternative - defaulting an unrecognised ref to "runs anywhere" - ships
    an unknown model to a researcher's CPU on the strength of a naming accident.
    """
    assert registry_model_id_for(_model("kraken://blla"), task=InferenceTask.segment) is None
    assert host_eligibility_for(None) is HostEligibility.remote
    assert eligible_targets_for_model(
        _model("kraken://blla", InferenceTask.segment), task=InferenceTask.segment
    ) == frozenset({CLOUD})


def test_an_id_the_registry_does_not_know_is_not_eligible_for_local() -> None:
    assert host_eligibility_for("not-a-real-model") is HostEligibility.remote


def test_the_shipped_registry_models_are_lite_and_may_run_on_either_host() -> None:
    """Reads the real ``registry.yaml``. If a model is ever marked ``remote``,
    this is where the change surfaces."""
    for registry_model_id in DEFAULT_REGISTRY_MODEL_IDS.values():
        assert host_eligibility_for(registry_model_id) is HostEligibility.local


@pytest.mark.parametrize(
    "registry_model_id",
    ["greek-calamari-v1", "armenian-calamari-v1", "syriac-calamari-v2", "coptic-calamari-v1"],
)
def test_every_transcribe_script_may_run_on_a_researchers_laptop(registry_model_id: str) -> None:
    """``DEFAULT_REGISTRY_MODEL_IDS`` names only one transcribe model, so the test
    above covers only one of the three scripts. The other two reach this module
    through a catalog ``artifact_ref`` instead, and a laptop-ineligible one would
    send a job to the cloud with no visible cause."""
    assert host_eligibility_for(registry_model_id) is HostEligibility.local
