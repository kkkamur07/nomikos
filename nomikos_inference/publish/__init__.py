"""Publish checkpoints and datasets to the Hugging Face Hub.

Runtime `hf://` resolution does not live here: it is `nomikos_inference.hub`
(ADR 0002), because it is on the inference path and a runtime that cannot fetch
its own weights is not a runtime. What lives here is publish-side and
repository-relative: the **Hub staging tree**, **Local bundled weights**, model
cards, and collection sync, none of which a researcher ever installs.

The directory sits inside `nomikos_inference/` so that it imports like
everything else in this repository, and it is named in the `exclude` list of
both build targets in `pyproject.toml` so that it still ships to nobody. Being
inside the package directory and being inside the wheel are two different
things, and the boundary ADR 0002 drew is the second one.
"""

from nomikos_inference.publish.client import (
    PublishClient,
    get_default_publish_client,
    set_default_publish_client,
    upload_enabled,
)
from nomikos_inference.publish.collection import CollectionSpec, load_collection_spec
from nomikos_inference.publish.dataset import (
    DatasetPublishPlan,
    plan_dataset_publish,
    publish_dataset,
)
from nomikos_inference.publish.model import ModelPublishPlan, plan_model_publish, publish_model
from nomikos_inference.publish.paths import (
    ARTIFACTS_ROOT,
    DEFAULT_COLLECTION_PATH,
    DEFAULT_LOCAL_BUNDLED_ROOT,
    DEFAULT_STAGING_ROOT,
    PUBLISH_ROOT,
)
from nomikos_inference.publish.staging import (
    DatasetStagingRef,
    ModelStagingRef,
    build_dataset_readme,
    build_model_card,
    dataset_staging_dir,
    hub_repo_slug,
    model_staging_dir,
    validate_dataset_slug,
    validate_dataset_staging,
    validate_model_staging,
)
from nomikos_inference.publish.sync import CollectionSyncPlan, plan_collection_sync, sync_collection

__all__ = [
    "ARTIFACTS_ROOT",
    "DEFAULT_COLLECTION_PATH",
    "DEFAULT_LOCAL_BUNDLED_ROOT",
    "DEFAULT_STAGING_ROOT",
    "PUBLISH_ROOT",
    "CollectionSpec",
    "CollectionSyncPlan",
    "DatasetPublishPlan",
    "DatasetStagingRef",
    "ModelPublishPlan",
    "ModelStagingRef",
    "PublishClient",
    "build_dataset_readme",
    "build_model_card",
    "dataset_staging_dir",
    "get_default_publish_client",
    "hub_repo_slug",
    "load_collection_spec",
    "model_staging_dir",
    "plan_collection_sync",
    "plan_dataset_publish",
    "plan_model_publish",
    "publish_dataset",
    "publish_model",
    "set_default_publish_client",
    "sync_collection",
    "upload_enabled",
    "validate_dataset_slug",
    "validate_dataset_staging",
    "validate_model_staging",
]
