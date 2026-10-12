import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { api, refreshAccessToken } from "../api/client";
import { clearLoginRedirectGuard } from "./session";
import { clearAccessToken, getAccessToken, setAccessToken } from "./storage";

type AuthStatus = "restoring" | "authenticated" | "anonymous";

type AuthContextValue = {
  status: AuthStatus;
  /**
   * Whether the signed-in account may open `/admin`; `null` until `/me` has
   * answered. Display only: the API checks `users.is_admin` on every admin route.
   */
  isAdmin: boolean | null;
  establish: (accessToken: string) => void;
  logout: () => Promise<void>;
};

const fallbackContext: AuthContextValue = {
  status: getAccessToken() ? "authenticated" : "anonymous",
  isAdmin: false,
  establish: (accessToken) => {
    setAccessToken(accessToken);
    clearLoginRedirectGuard();
  },
  logout: async () => {
    clearAccessToken();
  },
};

const AuthContext = createContext<AuthContextValue>(fallbackContext);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>(
    getAccessToken() ? "authenticated" : "restoring",
  );

  useEffect(() => {
    if (getAccessToken()) return;
    let active = true;
    void refreshAccessToken()
      .then(() => {
        if (!active) return;
        clearLoginRedirectGuard();
        setStatus("authenticated");
      })
      .catch(() => {
        if (!active) return;
        clearAccessToken();
        setStatus("anonymous");
      });
    return () => {
      active = false;
    };
  }, []);

  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);

  useEffect(() => {
    if (status !== "authenticated") {
      setIsAdmin(status === "anonymous" ? false : null);
      return;
    }
    const token = getAccessToken();
    if (!token) {
      setIsAdmin(false);
      return;
    }
    let active = true;
    setIsAdmin(null);
    // First without auth recovery, wrapped so a synchronous throw lands in the same
    // catch. A failure is not an answer (a network blip, a 5xx, a token that expired
    // since establish), so retry once with the normal recovery. If that fails too,
    // `isAdmin` stays null: the link stays hidden and /admin keeps loading rather
    // than turning a real admin away as not found.
    void Promise.resolve()
      .then(() => api.meWithoutRecovery(token))
      .catch(() => api.me())
      .then((user) => {
        if (active) setIsAdmin(user?.is_admin === true);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [status]);

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      isAdmin,
      establish: (accessToken) => {
        setAccessToken(accessToken);
        clearLoginRedirectGuard();
        setStatus("authenticated");
      },
      logout: async () => {
        try {
          await api.logout();
        } finally {
          clearAccessToken();
          setStatus("anonymous");
        }
      },
    }),
    [status, isAdmin],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuthSession(): AuthContextValue {
  return useContext(AuthContext);
}
