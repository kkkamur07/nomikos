-- 2026-09-20: register the 0.4.0 models and show the segmenters as "kraken" and "ppocr".
--
-- The picker prints inference_models.name. Dispatch never reads name: it parses
-- artifact_ref (registry://<registry id>?tag=stable), so the names are free to change
-- while the registry ids (blla-segment, ppocr-segment) stay what the worker knows.
-- A second segment row never changes the default: defaults come from model_bindings,
-- and a job without model_id falls back to the registry id blla-segment.
-- Safe to run twice.
begin;

-- kraken blla segmenter: display name only.
update inference_models set name = 'kraken'
 where name = 'blla-segment'
   and not exists (select 1 from inference_models where name = 'kraken');

-- PP-OCRv6 segmenter: rename if it was already inserted under its registry id, else insert.
update inference_models set name = 'ppocr'
 where name = 'ppocr-segment'
   and not exists (select 1 from inference_models where name = 'ppocr');

-- noise_policy: "flag" keeps suspect detections, marks them and reads them last;
-- "drop" removes them (precision 0.916 to 0.962 on the 12 Coptic pages).
insert into inference_models (id, name, provider, task, artifact_ref, default_params)
values (gen_random_uuid(), 'ppocr', 'ppocr', 'segment',
        'registry://ppocr-segment?tag=stable',
        '{"device": "cpu", "noise_policy": "flag"}')
on conflict (name) do nothing;

-- Syriac PP-OCRv6 recognizer.
insert into inference_models (id, name, provider, task, artifact_ref, default_params)
values (gen_random_uuid(), 'syriac-ppocr-v1', 'ppocr', 'transcribe',
        'registry://syriac-ppocr-v1?tag=stable',
        '{"device": "cpu"}')
on conflict (name) do nothing;

commit;
