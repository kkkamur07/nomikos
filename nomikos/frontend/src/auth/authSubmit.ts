import { toast } from "../components/ui/toast";
import type { TokenResponse } from "../api/client";

/**
 * The submit flow login and registration share: loading flag, token
 * handshake, success toast, redirect, failure toast. The request, the copy
 * and the failure mapping stay with the caller - a sign-in failure reads as
 * "Sign in failed" while a registration failure carries the API's message.
 */
export async function submitAuthForm(options: {
  setLoading: (loading: boolean) => void;
  request: () => Promise<TokenResponse>;
  establish: (token: string) => void;
  successMessage: string;
  failureMessage: (err: unknown) => string;
  redirect: () => void;
}): Promise<void> {
  options.setLoading(true);
  try {
    const token = await options.request();
    options.establish(token.access_token);
    toast.success(options.successMessage);
    options.redirect();
  } catch (err) {
    toast.error(options.failureMessage(err));
  } finally {
    options.setLoading(false);
  }
}
