import { useCallback, useRef, useState, type SetStateAction } from "react";
import type { InferenceModelResponse } from "../../../api/client";

/**
 * The two model pickers' catalogs and choices. `fetchPartContent` fills them
 * from the catalog and the page's bindings.
 */
export function useEditorModels() {
  const [transcribeModels, setTranscribeModels] = useState<
    InferenceModelResponse[]
  >([]);
  const [selectedTranscribeModelId, setSelectedTranscribeModelId] = useState<
    string | null
  >(null);
  const [segmentModels, setSegmentModels] = useState<InferenceModelResponse[]>(
    [],
  );
  /**
   * One of the `segmentModels` ids once the catalog loads: the picker has
   * no empty entry, so the segment request always carries it. Null only
   * while the catalog is empty or failed to load, when the select is
   * disabled and `model_id` is not sent. A page turn keeps an explicit
   * choice (see fetchPartContent).
   */
  const [selectedSegmentModelId, setSelectedSegmentModelId] = useState<
    string | null
  >(null);

  /**
   * Whether the segment picker choice came from the user. Set by the
   * picker's change handler, cleared on a document-level load. While set,
   * page turns keep the choice instead of applying the next part's binding.
   */
  const segmentChoiceIsExplicitRef = useRef(false);
  const chooseSegmentModelId = useCallback(
    (value: SetStateAction<string | null>) => {
      segmentChoiceIsExplicitRef.current = true;
      setSelectedSegmentModelId(value);
    },
    [],
  );
  const segmentChoiceIsExplicit = useCallback(
    () => segmentChoiceIsExplicitRef.current,
    [],
  );

  /**
   * Whether the transcribe picker choice came from the user. Same contract as
   * the segment one above: set by the picker's change handler, cleared on a
   * document-level load, read when a read lands so an in-flight choice wins.
   */
  const transcribeChoiceIsExplicitRef = useRef(false);
  const chooseTranscribeModelId = useCallback(
    (value: SetStateAction<string | null>) => {
      transcribeChoiceIsExplicitRef.current = true;
      setSelectedTranscribeModelId(value);
    },
    [],
  );
  const transcribeChoiceIsExplicit = useCallback(
    () => transcribeChoiceIsExplicitRef.current,
    [],
  );
  const forgetModelChoices = useCallback(() => {
    segmentChoiceIsExplicitRef.current = false;
    transcribeChoiceIsExplicitRef.current = false;
  }, []);

  return {
    transcribeModels,
    setTranscribeModels,
    selectedTranscribeModelId,
    setSelectedTranscribeModelId,
    chooseTranscribeModelId,
    transcribeChoiceIsExplicit,
    segmentModels,
    setSegmentModels,
    selectedSegmentModelId,
    setSelectedSegmentModelId,
    chooseSegmentModelId,
    segmentChoiceIsExplicit,
    forgetModelChoices,
  };
}
