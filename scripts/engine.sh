# Container and volume ownership checks shared by the Linux lifecycle scripts.
# shellcheck shell=sh

name="vllm-zero-to-hero-pt3"
volume="vllm-zero-to-hero-pt3-models"
managed_label="io.github.red-hat-ai-dev.vllm-zero-to-hero-pt3.managed"

container_label() {
  "$1" container inspect --format "{{ index .Config.Labels \"$3\" }}" "$2"
}

volume_id() {
  "$1" volume inspect --format "{{ index .Labels \"${3%.managed}.volume-id\" }}" "$2"
}

new_resource_id() {
  od -An -N16 -tx1 /dev/urandom | tr -d '[:space:]'
}

require_owned_volume() {
  volume_owner=$("$1" volume inspect --format "{{ index .Labels \"$3\" }}" "$2") || return 1
  if [ "$volume_owner" != true ]; then
    echo "Error: refusing to use or remove volume '$2' in $1 (missing lesson label)." >&2
    return 1
  fi
}

ensure_model_volume() {
  if ! "$engine" volume inspect "$volume" >/dev/null 2>&1; then
    new_id=$(new_resource_id)
    [ -n "$new_id" ] || return 1
    "$engine" volume create --label "$managed_label=true" \
      --label "${managed_label%.managed}.volume-id=$new_id" "$volume" >/dev/null || return 1
  fi
  # A concurrent create can return an existing volume without changing its labels.
  require_owned_volume "$engine" "$volume" "$managed_label"
}

remove_model_volume() {
  [ -n "$managed_volume_path" ] || return 0
  if "$engine" volume inspect "$volume" >/dev/null 2>&1; then
    require_owned_volume "$engine" "$volume" "$managed_label" || return 1
    current_id=$(volume_id "$engine" "$volume" "$managed_label") || return 1
    if [ "$current_id" != "$managed_volume_id" ]; then
      echo "Error: volume '$volume' changed during cleanup; leaving it in place." >&2
      return 1
    fi
    echo "Removing lesson model volume '$volume' with $engine..."
    "$engine" volume rm "$volume" >/dev/null
  fi
}

stop_lesson_container() {
  if [ -z "$managed_container_id" ]; then
    echo "vLLM is already stopped."
    return
  fi
  echo "Stopping the vLLM container..."
  "$engine" rm --force "$managed_container_id" >/dev/null || return 1
  echo "vLLM container stopped."
}

remove_launched_container() {
  # A failed run can leave a partial container; an overlapping launcher can
  # also win the name race. Check this attempt's label on an immutable ID.
  [ -n "${launch_id:-}" ] || return 0
  attempt_id="${container_id:-}"
  if [ -z "$attempt_id" ]; then
    attempt_id=$("$engine" container inspect --format '{{.Id}}' "$name" 2>/dev/null || true)
  fi
  if [ -n "$attempt_id" ] && [ "$(container_label "$engine" "$attempt_id" "$launch_key" 2>/dev/null)" = "$launch_id" ]; then
    "$engine" rm --force "$attempt_id" >/dev/null
  fi
}

start_lesson_container() {
  launch_key="${managed_label%.managed}.launch"
  launch_id=$(new_resource_id)
  [ -n "$launch_id" ] || return 1
  expected_volume_id=$(volume_id "$engine" "$volume" "$managed_label") || return 1
  if ! container_id=$("$engine" run -d --name "$name" --label "$launch_key=$launch_id" "$@"); then
    container_id=""
    remove_launched_container
    return 1
  fi
  # A simultaneous cleanup may remove the volume just before run mounts it.
  # Engines can implicitly recreate it without our ownership labels.
  if [ -z "$container_id" ] || ! require_owned_volume "$engine" "$volume" "$managed_label" ||
      [ "$(volume_id "$engine" "$volume" "$managed_label")" != "$expected_volume_id" ]; then
    echo "Error: the lesson container or model volume changed during startup." >&2
    remove_launched_container
    return 1
  fi
}

# Resolve a reachable engine using this lesson's labeled container or volume.
# Return both the engine and the verified container ID, safe across name reuse.
resolve_managed_engine() {
  resource_name="$1"
  resource_label="$2"
  resource_volume="${3:-}"
  engine=""
  managed_container_id=""
  managed_volume_path=""
  managed_volume_id=""
  resource_key=""
  unavailable=""
  if [ -n "${ENGINE:-}" ]; then set -- "$ENGINE"; else set -- podman docker; fi
  for candidate in "$@"; do
    command -v "$candidate" >/dev/null 2>&1 || {
      if [ -n "${ENGINE:-}" ]; then
        echo "Error: ENGINE command not found: $candidate" >&2
        return 1
      fi
      continue
    }
    if ! "$candidate" info >/dev/null 2>&1; then
      unavailable="$unavailable $candidate"
      continue
    fi
    candidate_id=$("$candidate" container inspect --format '{{.Id}}' "$resource_name" 2>/dev/null || true)
    if [ -n "$candidate_id" ]; then
      owner=$(container_label "$candidate" "$candidate_id" "$resource_label") || return 1
      if [ "$owner" != true ]; then
        echo "Error: refusing to remove unrelated container '$resource_name' in $candidate (missing lesson label)." >&2
        return 1
      fi
    fi
    volume_path=""
    candidate_volume_id=""
    if [ -n "$resource_volume" ]; then
      volume_path=$("$candidate" volume inspect --format '{{.Mountpoint}}' "$resource_volume" 2>/dev/null || true)
      if [ -n "$volume_path" ]; then
        require_owned_volume "$candidate" "$resource_volume" "$resource_label" || return 1
        candidate_volume_id=$(volume_id "$candidate" "$resource_volume" "$resource_label") || return 1
      fi
    fi
    [ -n "$candidate_id$volume_path" ] || continue
    # A docker command backed by Podman exposes the same resources twice.
    key="$candidate_id:$volume_path:$candidate_volume_id"
    # Older hand-labeled volumes lack an instance ID. Equal mount paths alone
    # cannot establish that two reachable Docker/Podman daemons are aliases.
    if [ -z "$candidate_id" ]; then
      case "$candidate_volume_id" in ""|"<no value>") key="$candidate:$key" ;; esac
    fi
    if [ -n "$engine" ] && [ "$key" != "$resource_key" ]; then
      echo "Error: tutorial resources exist in both Podman and Docker. Set ENGINE explicitly." >&2
      return 1
    fi
    engine="$candidate"
    managed_container_id="$candidate_id"
    managed_volume_path="$volume_path"
    managed_volume_id="$candidate_volume_id"
    resource_key="$key"
  done
  if [ -z "$engine" ] && [ -n "$unavailable" ]; then
    echo "Error: cannot inspect container engine(s):$unavailable. Start them or set ENGINE to the owner." >&2
    return 1
  fi
}
