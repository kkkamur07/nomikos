/**
 * What the interface knows about **inference host**s.
 *
 * There is no client here, and there is deliberately nothing to add one to.
 * Since ADR 0002 the browser never reaches an **inference agent**: it reads an
 * account setting, it reads **capacity**, and it reads what each job says about
 * where it ran. Everything else happens between the agent and the platform.
 *
 * Only the names imported through this barrel stay here
 * (`submissionRefusalExplanation`, `useHostPreference`); everything else is
 * imported from its submodule directly.
 */
export { useHostPreference } from "./hostPreference";
export { submissionRefusalExplanation } from "./executionTarget";
