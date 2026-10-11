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
    // No auth recovery, and wrapped so a synchronous throw lands in the same
    // catch: a failed lookup only hides the admin link, it never signs anyone out.
    void Promise.resolve()
      .then(() => api.meWithoutRecovery(token))
      .then((user) => {
        if (active) setIsAdmin(user?.is_admin === true);
      })
      .catch(() => {
        if (active) setIsAdmin(false);
      });
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
