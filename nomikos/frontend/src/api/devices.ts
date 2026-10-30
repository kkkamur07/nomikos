/**
 * The browser's half of device pairing.
 *
 * Every route is Bearer-authenticated and none requires the CSRF header, which
 * `apiRequest` adds anyway when the cookie is present and the server ignores.
 *
 * `verification_token` travels in a POST body on all three pairing routes,
 * including the lookup, so the consent token never reaches a path, a query
 * string or a server log. See `components/devices/pairingToken.ts` for the other
 * end of that promise.
 */
import { apiRequest } from "./http";
import type { components } from "./schema";

export type DeviceResponse = components["schemas"]["DeviceResponse"];
export type DeviceStatus = components["schemas"]["DeviceStatus"];
export type PairingRequestResponse =
  components["schemas"]["PairingRequestResponse"];
type PairingLookupRequest = components["schemas"]["PairingLookupRequest"];
type PairingConsentRequest = components["schemas"]["PairingConsentRequest"];

export const devicesApi = {
  /**
   * Resolve the fragment token into the consent screen's contents.
   *
   * Unknown, expired, consumed and denied are one indistinguishable 404 by
   * design; the caller must not try to tell them apart.
   */
  lookupPairing: (verificationToken: string): Promise<PairingRequestResponse> =>
    apiRequest<PairingRequestResponse>("/devices/pairings/lookup", {
      method: "POST",
      body: {
        verification_token: verificationToken,
      } satisfies PairingLookupRequest,
    }),

  approvePairing: (
    pairingId: string,
    verificationToken: string,
  ): Promise<DeviceResponse> =>
    apiRequest<DeviceResponse>(
      `/devices/pairings/${encodeURIComponent(pairingId)}/approve`,
      {
        method: "POST",
        body: {
          verification_token: verificationToken,
        } satisfies PairingConsentRequest,
      },
    ),

  denyPairing: (pairingId: string, verificationToken: string): Promise<void> =>
    apiRequest<void>(
      `/devices/pairings/${encodeURIComponent(pairingId)}/deny`,
      {
        method: "POST",
        body: {
          verification_token: verificationToken,
        } satisfies PairingConsentRequest,
      },
    ),

  listDevices: (includeRevoked: boolean): Promise<DeviceResponse[]> =>
    apiRequest<DeviceResponse[]>(
      includeRevoked ? "/devices?include_revoked=true" : "/devices",
    ),

  revokeDevice: (deviceId: string): Promise<void> =>
    apiRequest<void>(`/devices/${encodeURIComponent(deviceId)}`, {
      method: "DELETE",
    }),
} as const;
