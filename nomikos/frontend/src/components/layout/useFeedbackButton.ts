import { useEffect, useRef, useState } from "react";

/** Open/thanks state for a Feedback button, shared by TopNav and the editor. */
export function useFeedbackButton() {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [thanks, setThanks] = useState(false);
  const thanksTimer = useRef<number | undefined>(undefined);

  useEffect(() => () => clearTimeout(thanksTimer.current), []);

  const handleSent = () => {
    setDialogOpen(false);
    setThanks(true);
    clearTimeout(thanksTimer.current);
    thanksTimer.current = window.setTimeout(() => setThanks(false), 2000);
  };

  return {
    dialogOpen,
    thanks,
    openDialog: () => setDialogOpen(true),
    closeDialog: () => setDialogOpen(false),
    handleSent,
  };
}
