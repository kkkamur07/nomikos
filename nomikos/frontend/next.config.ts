import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";
import { securityHeadersFor } from "./securityHeaders";

const platformApi = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");
// A relative value (same-origin API) has no origin of its own: 'self' covers it,
// and `new URL` would otherwise throw while the config loads.
function originOf(url: string): string {
  try {
    return new URL(url).origin;
  } catch {
    return "'self'";
  }
}
const platformApiOrigin = originOf(platformApi);
const frontendRoot = fileURLToPath(new URL(".", import.meta.url));

const nextConfig: NextConfig = {
  // Standalone exists for `nomikos/frontend/Dockerfile`, which copies
  // `.next/standalone`. Vercel traces the server itself and its onBuildComplete
  // hook reads `.next/next-server.js.nft.json`, which the Turbopack standalone
  // build does not emit as of Next 16.3.0 - the deploy fails on ENOENT. Leave
  // `output` unset there so Vercel takes its own supported path.
  output: process.env.VERCEL ? undefined : "standalone",
  // Keep legacy route components in src/pages while App Router owns routing.
  pageExtensions: ["next.tsx", "next.ts", "next.jsx", "next.js"],
  turbopack: {
    root: frontendRoot,
  },
  // The dev badge covered the editor's bottom corners (island, strip Save).
  devIndicators: false,
  async headers() {
    // On Vercel these come from vercel.json; returning them here too would set
    // each header twice (two CSPs, both enforced). Apply only off-Vercel.
    if (process.env.VERCEL) return [];
    return [
      { source: "/:path*", headers: securityHeadersFor(platformApiOrigin) },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/media/:path*",
        destination: `${platformApi}/media/:path*`,
      },
    ];
  },
};

export default nextConfig;
