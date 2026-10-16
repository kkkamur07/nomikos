/**
 * How a researcher installs and starts the **inference agent**.
 *
 * These are commands, not download links (ADR 0002): a per-OS installer URL
 * pointing at `releases/latest/download/…` breaks the moment the release
 * workflow that built it changes, while a command a researcher types cannot
 * rot the same way. There is one set of them for every platform because
 * there is one **published package**.
 *
 * Nothing here is fetched by the browser. The page says what to run and then
 * stops: the agent talks to the platform outbound, and this tab learns it is
 * running from **capacity** on the account's execution-target response, never
 * by probing the machine it happens to be displayed on.
 */

/** The one distribution. A hosted worker installs the same one (ADR 0002). */
const AGENT_PACKAGE_NAME = "nomikos-inference";

/**
 * `--torch-backend=cpu` is load-bearing, not a tuning flag: without it a plain
 * resolve drags sixteen CUDA wheels behind Torch on Linux and Windows, which is
 * most of the download and useless on a laptop. It needs uv >= 0.10.
 */
export const AGENT_INSTALL_COMMAND = `uv tool install ${AGENT_PACKAGE_NAME} --torch-backend=cpu`;

/** For an environment that already has pip and no uv. */
export const AGENT_INSTALL_COMMAND_PIP = `pip install ${AGENT_PACKAGE_NAME}`;

/** Links this machine to the account. Prints a **confirmation code** to compare. */
export const AGENT_PAIR_COMMAND = "nomikos pair";

/** Starts claiming. Nothing runs on this account's pages until it does. */
export const AGENT_RUN_COMMAND = "nomikos run";
