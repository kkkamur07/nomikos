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
   * Null means "Default": `model_id` is not sent and the backend resolves
   * the binding or the worker's own default, exactly as before this picker
   * existed. It is never initialised from `segmentModels[0]`, and a page
   * turn keeps an explicit choice (see fetchPartContent).
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
  const forgetSegmentChoice = useCallback(() => {
    segmentChoiceIsExplicitRef.current = false;
  }, []);

  return {
    transcribeModels,
    setTranscribeModels,
    selectedTranscribeModelId,
    setSelectedTranscribeModelId,
    segmentModels,
    setSegmentModels,
    selectedSegmentModelId,
    setSelectedSegmentModelId,
    chooseSegmentModelId,
    segmentChoiceIsExplicit,
    forgetSegmentChoice,
  };
}
