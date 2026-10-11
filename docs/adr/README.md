# Architecture Decision Records

One file per decision, named `NNNN-kebab-case-title.md`, numbered in the order the decision
was taken. A record is written when a choice is expensive to reverse: schema shape, credential
design, transport direction, trust boundaries.

Records are immutable once merged. A decision that is later reversed gets a new record that
supersedes the old one, and the old one gains a `Superseded by` line. Do not edit history. The
value of an ADR is that it says what was believed at the time, and why.

| #                                                                    | Title                                                          | Status                                                       |
| -------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------ |
| [0001](0001-outbound-helper-device-pairing.md)                       | Outbound helper device pairing and device-scoped tokens        | Accepted                                                     |
| [0002](0002-inference-cli-replaces-loopback-helper.md)               | Local inference is a CLI, not a loopback service               | Accepted; its packaging section, amended by 0004, restored by 0006 |
| [0003](0003-single-job-queue-cloud-worker-claims-like-a-device.md)   | One job queue: the cloud worker claims like any paired device  | Accepted                                                     |
| [0004](0004-pytorch-is-the-inference-runtime.md)                     | PyTorch is the inference runtime; ONNX is archived             | Superseded by 0006                                           |
| [0005](0005-agent-claim-endpoint-and-the-inference-service-account.md) | The agent claim endpoint and the inference service account     | Accepted                                                     |
| [0006](0006-onnx-runtime-is-the-inference-runtime.md)                | ONNX Runtime is the inference runtime; PyTorch builds the artifact | Accepted                                                 |
| [0007](0007-serving-preprocessing-reproduces-the-training-loader.md) | Serving preprocessing reproduces the training loader, exactly  | Accepted                                                     |
| [0008](0008-server-owned-reading-order-for-lines.md)               | Server-owned reading order for lines                           | Accepted                                                     |
| [0009](0009-script-aware-document-exports.md)                         | Script-aware document exports (docx and shaped PDF)            | Accepted                                                     |
