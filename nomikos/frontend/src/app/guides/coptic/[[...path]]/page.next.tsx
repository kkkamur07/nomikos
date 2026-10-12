import { notFound } from "next/navigation";

// The Coptic guide was retired. Its old URLs answer a real 404 rather than
// falling into the catch-all, which redirects to sign-in with a soft 200.
export default function RetiredCopticGuidePage() {
  notFound();
}
