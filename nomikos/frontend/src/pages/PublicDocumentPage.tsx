import {
  PublicDocumentContent,
  PublicReaderHeader,
  PublicReaderNav,
} from "../components/public/PublicDocumentSections";
import { usePublicDocument } from "./usePublicDocument";

export function PublicDocumentPage() {
  const model = usePublicDocument();
  return (
    <div className="page page--public">
      <PublicReaderNav {...model} />

      <PublicReaderHeader {...model} />

      <main className="pub-workspace content-wrap">
        <PublicDocumentContent {...model} />
      </main>
    </div>
  );
}
