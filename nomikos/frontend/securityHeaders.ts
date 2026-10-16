// Kept in sync with vercel.json. Vercel serves those at the edge; the self-hosted
// `node server.js` runtime never reads vercel.json, so the same headers are applied
// from next.config.ts for the Docker deployment. `securityHeaders.test.ts` fails
// if the two drift apart.
export const securityHeaders = [
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
    value:
      "default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob: https://api.nomikos.app; frame-src 'self' blob:; connect-src 'self' https://api.nomikos.app https://mknnoqpavpmxsyctwjdt.supabase.co; worker-src 'self' blob:",
  },
];
