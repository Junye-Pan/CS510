from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DockerMount:
    source: Path
    target: str
    read_only: bool = False


class DockerNetworkPolicyError(ValueError):
    def __init__(self, reason: str, message: str, *, enforcement: dict[str, Any]) -> None:
        super().__init__(message)
        self.reason = reason
        self.enforcement = enforcement


def build_docker_run_command(
    *,
    image: str,
    command: str | list[Any],
    mounts: list[DockerMount],
    workdir: str,
    network_policy: dict[str, Any],
    requested_network_mode: str | None = None,
    requires_control_plane: bool = False,
    control_plane_relay_socket: Path | None = None,
    control_plane_relay_url: str | None = None,
    container_control_plane_socket_path: str = "/ao-control/control.sock",
    env: dict[str, Any] | None = None,
    container_name: str | None = None,
    cap_drop_all: bool = True,
    no_new_privileges: bool = True,
    pids_limit: int | None = None,
    memory: str | None = None,
    cpus: str | None = None,
    add_hosts: list[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    external = str(network_policy.get("external_internet") or "allow")
    control_plane = str(network_policy.get("control_plane") or "allow")
    network_mode = requested_network_mode
    relay_configured = control_plane_relay_socket is not None or bool(control_plane_relay_url)
    tcp_relay_configured = bool(control_plane_relay_url) and control_plane_relay_socket is None
    outbound_proxy_url = str(network_policy.get("outbound_proxy_url") or "")
    outbound_proxy_socket = network_policy.get("outbound_proxy_socket")
    outbound_proxy_container_socket = str(network_policy.get("outbound_proxy_container_socket") or "/ao-network/proxy.sock")
    outbound_proxy_bridge_port = int(network_policy.get("outbound_proxy_bridge_port") or 8765)
    outbound_proxy_no_proxy = str(network_policy.get("outbound_proxy_no_proxy") or "127.0.0.1,localhost")
    policy_weakened = False
    unix_proxy_configured = bool(outbound_proxy_socket)
    if unix_proxy_configured:
        outbound_proxy_url = f"http://127.0.0.1:{outbound_proxy_bridge_port}"

    if requires_control_plane and control_plane != "allow":
        enforcement = {
            "external_internet": external,
            "control_plane": control_plane,
            "requested_network_mode": requested_network_mode,
            "docker_network_mode": network_mode or "default",
            "external_internet_enforced": external == "deny" and network_mode == "none",
            "control_plane_available": False,
            "control_plane_requires_relay": False,
            "control_plane_relay_configured": relay_configured,
        }
        raise DockerNetworkPolicyError(
            "docker_control_plane_denied_by_policy",
            "Docker worker requires control-plane access, but network policy denies control_plane",
            enforcement=enforcement,
        )

    if external == "deny":
        if requested_network_mode and requested_network_mode != "none" and not (
            tcp_relay_configured and requested_network_mode in {"bridge", "default"}
        ):
            enforcement = {
                "external_internet": external,
                "control_plane": control_plane,
                "requested_network_mode": requested_network_mode,
                "docker_network_mode": None,
                "external_internet_enforced": False,
                "control_plane_available": False,
            }
            raise DockerNetworkPolicyError(
                "docker_network_mode_violates_external_deny",
                f"external_internet=deny requires Docker network_mode=none, got {requested_network_mode!r}",
                enforcement=enforcement,
            )
        if requires_control_plane and control_plane == "allow" and not relay_configured:
            enforcement = {
                "external_internet": external,
                "control_plane": control_plane,
                "requested_network_mode": requested_network_mode,
                "docker_network_mode": "none",
                "external_internet_enforced": True,
                "control_plane_available": False,
                "control_plane_requires_relay": True,
                "control_plane_relay_configured": False,
            }
            raise DockerNetworkPolicyError(
                "docker_control_plane_relay_required",
                "Docker --network none enforces external_internet=deny but cannot reach the control plane; use a control-plane relay provider",
                enforcement=enforcement,
            )
        network_mode = "none"
    elif external == "audit" and unix_proxy_configured:
        if tcp_relay_configured and requires_control_plane:
            enforcement = {
                "external_internet": external,
                "control_plane": control_plane,
                "requested_network_mode": requested_network_mode,
                "docker_network_mode": "none",
                "external_internet_enforced": True,
                "control_plane_available": False,
                "control_plane_requires_relay": True,
                "control_plane_relay_configured": True,
                "control_plane_relay_transport": "tcp",
            }
            raise DockerNetworkPolicyError(
                "docker_tcp_relay_incompatible_with_proxy_isolation",
                "Docker proxy-only audit egress uses --network none; a TCP control-plane relay would be unreachable. Use the Unix-socket control-plane relay or an explicit TCP fallback without proxy-only isolation.",
                enforcement=enforcement,
            )
        if requested_network_mode and requested_network_mode != "none":
            enforcement = {
                "external_internet": external,
                "control_plane": control_plane,
                "requested_network_mode": requested_network_mode,
                "docker_network_mode": None,
                "external_internet_enforced": False,
                "control_plane_available": False,
            }
            raise DockerNetworkPolicyError(
                "docker_network_mode_violates_audit_proxy_isolation",
                f"external_internet=audit with a Unix-socket outbound proxy requires Docker network_mode=none, got {requested_network_mode!r}",
                enforcement=enforcement,
            )
        if requires_control_plane and control_plane == "allow" and not relay_configured:
            enforcement = {
                "external_internet": external,
                "control_plane": control_plane,
                "requested_network_mode": requested_network_mode,
                "docker_network_mode": "none",
                "external_internet_enforced": True,
                "control_plane_available": False,
                "control_plane_requires_relay": True,
                "control_plane_relay_configured": False,
            }
            raise DockerNetworkPolicyError(
                "docker_control_plane_relay_required",
                "Docker --network none enforces proxy-only audit egress but cannot reach the control plane; use a control-plane relay provider",
                enforcement=enforcement,
            )
        network_mode = "none"
    if external == "deny" and tcp_relay_configured and requires_control_plane:
        # TCP relay fallback is useful for Docker Desktop/Colima environments
        # where bind-mounted AF_UNIX sockets are unreliable, but it cannot be
        # combined with Docker's --network none. Keep the run explicit.
        network_mode = requested_network_mode or "bridge"
        policy_weakened = True

    enforcement = {
        "external_internet": external,
        "control_plane": control_plane,
        "requested_network_mode": requested_network_mode,
        "docker_network_mode": network_mode or "default",
        "external_internet_enforced": external in {"deny", "audit"} and network_mode == "none",
        "control_plane_available": control_plane == "allow" and (network_mode != "none" or relay_configured),
        "control_plane_requires_relay": external == "deny" and requires_control_plane and control_plane == "allow"
        or (external == "audit" and unix_proxy_configured and requires_control_plane and control_plane == "allow"),
        "control_plane_relay_configured": relay_configured,
        "control_plane_relay_transport": "tcp" if tcp_relay_configured else "unix-socket" if control_plane_relay_socket is not None else None,
        "outbound_audit_proxy_configured": bool(outbound_proxy_url),
        "outbound_audit_proxy_url": outbound_proxy_url or None,
        "outbound_audit_proxy_transport": "unix-socket" if unix_proxy_configured else "tcp" if outbound_proxy_url else None,
        "outbound_audit_mode": "unix_proxy_bridge" if unix_proxy_configured else "env_proxy" if outbound_proxy_url else None,
        "direct_network_disabled": network_mode == "none",
        "policy_weakened": policy_weakened,
        "policy_weakened_reason": "tcp_control_plane_relay_requires_docker_network" if policy_weakened else None,
    }
    docker_command: list[str] = ["docker", "run", "--rm"]
    if container_name:
        docker_command.extend(["--name", container_name])
    if cap_drop_all:
        docker_command.extend(["--cap-drop", "ALL"])
    if no_new_privileges:
        docker_command.extend(["--security-opt", "no-new-privileges"])
    if pids_limit is not None:
        docker_command.extend(["--pids-limit", str(pids_limit)])
    if memory:
        docker_command.extend(["--memory", memory])
    if cpus:
        docker_command.extend(["--cpus", cpus])
    if network_mode:
        docker_command.extend(["--network", network_mode])
    for host_entry in add_hosts or []:
        docker_command.extend(["--add-host", str(host_entry)])
    relay_env_names: set[str] = set()
    if control_plane_relay_socket is not None:
        docker_command.extend(
            [
                "-v",
                _volume_arg(
                    DockerMount(
                        source=control_plane_relay_socket.resolve(),
                        target=container_control_plane_socket_path,
                    )
                ),
                "-e",
                f"AO_CONTROL_API_URL=unix://{container_control_plane_socket_path}",
            ]
        )
        relay_env_names.add("AO_CONTROL_API_URL")
    elif control_plane_relay_url:
        docker_command.extend(["-e", f"AO_CONTROL_API_URL={control_plane_relay_url}"])
        relay_env_names.add("AO_CONTROL_API_URL")
    docker_env = dict(env or {})
    if unix_proxy_configured:
        docker_command.extend(
            [
                "-v",
                _volume_arg(
                    DockerMount(
                        source=Path(str(outbound_proxy_socket)).resolve(),
                        target=outbound_proxy_container_socket,
                    )
                ),
            ]
        )
        docker_env.setdefault("AO_OUTBOUND_PROXY_SOCKET", outbound_proxy_container_socket)
        docker_env.setdefault("AO_OUTBOUND_PROXY_BRIDGE_PORT", str(outbound_proxy_bridge_port))
    if outbound_proxy_url:
        docker_env.setdefault("HTTP_PROXY", outbound_proxy_url)
        docker_env.setdefault("HTTPS_PROXY", outbound_proxy_url)
        docker_env.setdefault("ALL_PROXY", outbound_proxy_url)
        docker_env.setdefault("NO_PROXY", outbound_proxy_no_proxy)
        docker_env.setdefault("http_proxy", outbound_proxy_url)
        docker_env.setdefault("https_proxy", outbound_proxy_url)
        docker_env.setdefault("all_proxy", outbound_proxy_url)
        docker_env.setdefault("no_proxy", outbound_proxy_no_proxy)
    for key, value in docker_env.items():
        if key in relay_env_names:
            continue
        docker_command.extend(["-e", f"{key}={value}"])
    for mount in mounts:
        docker_command.extend(["-v", _volume_arg(mount)])
    docker_command.extend(["-w", workdir, str(image)])
    if isinstance(command, str):
        docker_command.extend(["sh", "-lc", command])
    else:
        docker_command.extend(str(item) for item in command)
    return docker_command, enforcement


def build_local_docker_command(
    *,
    image: str,
    command: str | list[Any],
    cwd: Path,
    network_policy: dict[str, Any],
    requested_network_mode: str | None = None,
    requires_control_plane: bool = False,
    control_plane_relay_socket: Path | None = None,
    control_plane_relay_url: str | None = None,
    container_control_plane_socket_path: str = "/ao-control/control.sock",
    mounts: list[DockerMount] | None = None,
    workdir: str = "/workspace",
    env: dict[str, Any] | None = None,
    container_name: str | None = None,
    pids_limit: int | None = None,
    memory: str | None = None,
    cpus: str | None = None,
    add_hosts: list[str] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    return build_docker_run_command(
        image=image,
        command=command,
        mounts=mounts or [DockerMount(source=cwd.resolve(), target="/workspace")],
        workdir=workdir,
        network_policy=network_policy,
        requested_network_mode=requested_network_mode,
        requires_control_plane=requires_control_plane,
        control_plane_relay_socket=control_plane_relay_socket,
        control_plane_relay_url=control_plane_relay_url,
        container_control_plane_socket_path=container_control_plane_socket_path,
        env=env,
        container_name=container_name,
        cap_drop_all=False,
        no_new_privileges=False,
        pids_limit=pids_limit,
        memory=memory,
        cpus=cpus,
        add_hosts=add_hosts,
    )


def _volume_arg(mount: DockerMount) -> str:
    suffix = ":ro" if mount.read_only else ""
    return f"{mount.source.resolve()}:{mount.target}{suffix}"
