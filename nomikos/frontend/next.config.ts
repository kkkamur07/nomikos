import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";

const platformApi = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/$/, "");
const platformApiOrigin = new URL(platformApi).origin;
const frontendRoot = fileURLToPath(new URL(".", import.meta.url));

// Kept in sync with vercel.json. Vercel serves those at the edge; the self-hosted
// `node server.js` runtime never reads vercel.json, so the same headers are applied
// here for the Docker deployment. The API origin comes from NEXT_PUBLIC_API_BASE_URL.
const securityHeaders = [
  {
    key: "Strict-Transport-Security",
    value: "max-age=31536000; includeSubDomains; preload",
  },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  {
    key: "Permissions-Policy",
    value: "camera=(), geolocation=(), microphone=(), payment=(), usb=()",
  },
  {
    key: "Content-Security-Policy",
    value: `default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob: ${platformApiOrigin}; frame-src 'self' blob:; connect-src 'self' ${platformApiOrigin}; worker-src 'self' blob:`,
  },
];

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
  async headers() {
    // On Vercel these come from vercel.json; returning them here too would set
    // each header twice (two CSPs, both enforced). Apply only off-Vercel.
    if (process.env.VERCEL) return [];
    return [{ source: "/:path*", headers: securityHeaders }];
  },
  async rewrites() {
    return [
      {
        source: "/media/:path*",
        destination: `${platformApi}/media/:path*`,
      },
      { source: "/guides/coptic", destination: "/guides/coptic/index.html" },
    ];
  },
};

export default nextConfig;
