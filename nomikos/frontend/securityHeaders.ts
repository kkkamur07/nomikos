// Two copies of the same headers, each winning in one place. On Vercel,
// vercel.json wins: next.config.ts `headers()` returns nothing there, and its CSP
// names https://api.nomikos.app by hand. Off Vercel (Docker, `node server.js`,
// which never reads vercel.json) these win, with the API origin taken from
// NEXT_PUBLIC_API_BASE_URL. `securityHeaders.test.ts` fails if the two drift apart.
export function securityHeadersFor(apiOrigin: string) {
  return [
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
      value: `default-src 'self'; base-uri 'self'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob: ${apiOrigin}; frame-src 'self' blob:; connect-src 'self' ${apiOrigin}; worker-src 'self' blob:`,
    },
  ];
}

/** The production headers, the ones vercel.json carries. */
export const securityHeaders = securityHeadersFor("https://api.nomikos.app");
