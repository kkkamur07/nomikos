import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type UserResponse } from "../api/client";
import { ApiError } from "../api/errors";
import {
  devicesApi,
  invalidateAfter,
  resourceTags,
  type DeviceResponse,
} from "../api/resources";
import { userFacingMessage } from "../api/userFacingError";
import {
  hasAccessToken,
  isUnauthorized,
  navigateToLogin,
} from "../auth/session";
import { AppPageShell } from "../components/layout/AppPageShell";
import { DevicesTable } from "../components/devices/DevicesTable";
import { toast } from "../components/ui/toast";
import { useServerQuery } from "../hooks/useServerQuery";

type DevicesPageData = {
  me: UserResponse;
  devices: DeviceResponse[];
};

/**
 * The whole device surface is behind `DEVICE_PAIRING_ENABLED`, and a disabled
 * flag 404s it - deliberately indistinguishable from never having been
 * deployed. On the *list* route that has only one meaning, because a device
 * list is otherwise never absent, only empty. So this page can say so plainly
 * where the consent screen may not.
 */
type DevicesError =
  { kind: "unavailable" } | { kind: "failed"; message: string };

function devicesQueryKey(
  signedIn: boolean,
  includeRevoked: boolean,
): readonly [string, boolean] | null {
  // `includeRevoked` is part of the identity of the read: the two variants are
  // different lists, and sharing a cache entry would show one for the other.
  return signedIn ? ["devices-page", includeRevoked] : null;
}

function DevicesErrorBanner({
  error,
}: {
  error: DevicesError;
}): React.ReactNode {
  if (error.kind === "unavailable") {
    return (
      <div className="notice-banner" role="alert">
        <strong>Device pairing is off on this deployment</strong>
        Nothing can be paired until an administrator enables it.
      </div>
    );
  }
  return (
    <div className="notice-banner" role="alert">
      <strong>Devices unavailable</strong>
      {error.message}
    </div>
  );
}

export function DevicesPage() {
  const router = useRouter();
  const [includeRevoked, setIncludeRevoked] = useState(false);
  const [revokingDeviceId, setRevokingDeviceId] = useState<string | null>(null);

  // Redirecting before any request goes out leaves the page in its loading
  // state rather than flashing an empty list on the way to login.
  const signedIn = hasAccessToken();
  useEffect(() => {
    if (!signedIn) navigateToLogin(router);
  }, [signedIn, router]);

  const { data, loading, error } = useServerQuery<
    DevicesPageData,
    DevicesError
  >({
    // `includeRevoked` is part of the identity of the read: the two variants are
    // different lists, and sharing a cache entry would show one for the other.
    key: devicesQueryKey(signedIn, includeRevoked),
    tags: [resourceTags.currentUser, resourceTags.devices],
    read: async () => {
      const [me, devices] = await Promise.all([
        api.me(),
        devicesApi.listDevices(includeRevoked),
      ]);
      return { me, devices };
    },
    onError: (err) => {
      if (isUnauthorized(err)) {
        navigateToLogin(router);
        return null;
      }
      if (err instanceof ApiError && err.status === 404) {
        return { kind: "unavailable" };
      }
      const message = userFacingMessage(err, "Failed to load your devices");
      toast.error(message);
      return { kind: "failed", message };
    },
  });

  const devices = data?.devices ?? [];
  const username = data?.me.username ?? null;

  const handleRevoke = async (deviceId: string) => {
    setRevokingDeviceId(deviceId);
    try {
      await devicesApi.revokeDevice(deviceId);
      toast.success("Device revoked");
      // Invalidation refetches the mounted list; no second refetch needed.
      await invalidateAfter.deviceListChanged();
    } catch (err) {
      toast.error(userFacingMessage(err, "Failed to revoke that device"));
    } finally {
      setRevokingDeviceId(null);
    }
  };

  return (
    <AppPageShell
      breadcrumb={[
        { label: "Settings", href: "/settings" },
        { label: "Devices" },
      ]}
      username={username}
      title="Devices"
      subtitle="Computers allowed to run jobs on your account"
      headerActions={
        <label className="field-check">
          <input
            type="checkbox"
            checked={includeRevoked}
            onChange={(event) => setIncludeRevoked(event.target.checked)}
          />
          Show revoked
        </label>
      }
    >
      {error && <DevicesErrorBanner error={error} />}

      <p className="section-label" id="devices-label">
        Paired devices
      </p>
      <DevicesTable
        id="devices-label"
        caption="Devices paired with your account"
        devices={devices}
        loading={loading && !error}
        emptyText={
          error
            ? "No devices to show"
            : "No paired devices. Run the helper on a computer to pair one."
        }
        onRevoke={(deviceId) => void handleRevoke(deviceId)}
        revokingDeviceId={revokingDeviceId}
      />

      <p className="list-hint">
        Revoking takes effect on that computer&rsquo;s next request. The helper
        is not consulted, so it works from a phone.
      </p>
    </AppPageShell>
  );
}
