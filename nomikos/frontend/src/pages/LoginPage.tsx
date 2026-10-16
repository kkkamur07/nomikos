import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { api } from "../api/client";
import { userFacingMessage } from "../api/userFacingError";
import { authRedirectTarget } from "../auth/redirect";
import { submitAuthForm } from "../auth/authSubmit";
import { useAuthSession } from "../auth/AuthProvider";
import { AuthFormWrap, AuthLayout } from "../components/layout/AuthLayout";
import { SessionRestoring } from "../components/SessionRestoring";
import { PasswordInput } from "../components/ui/PasswordInput";

export function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const from = authRedirectTarget(searchParams?.get("callbackUrl"));
  const { status, establish } = useAuthSession();

  useEffect(() => {
    if (status === "authenticated") {
      router.replace(from);
    }
  }, [from, router, status]);

  if (status === "restoring") {
    return <SessionRestoring />;
  }

  if (status === "authenticated") {
    return <SessionRestoring label="Taking you back…" />;
  }

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void submitAuthForm({
      setLoading,
      request: () => api.login({ email, password }),
      establish,
      successMessage: "Signed in",
      failureMessage: (err) => userFacingMessage(err, "Sign in failed"),
      redirect: () => router.replace(from),
    });
  };

  return (
    <AuthLayout
      headline="Nomikos platform for HTR"
      tagline="Segment pages, run multiple HTR models, and review transcriptions in one editor. Publish live public pages when a document is ready."
    >
      <AuthFormWrap>
        <h1>Sign in</h1>
        <p className="auth-sub">Your workspace</p>
        <form onSubmit={onSubmit}>
          <div className="field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              placeholder="you@institution.edu"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="password">Password</label>
            <PasswordInput
              id="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <button
            type="submit"
            className="btn btn-primary btn-block mt-4"
            disabled={loading}
          >
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <p className="auth-footer-link">
          No account?{" "}
          <Link href={`/register?callbackUrl=${encodeURIComponent(from)}`}>
            Register
          </Link>
        </p>
      </AuthFormWrap>
    </AuthLayout>
  );
}
