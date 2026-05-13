#!/bin/bash
# Save/load podman images to NFS so wrapper builds survive a node change.
#
# /tmp storage is node-local xfs (fast, supports xattrs, required by podman
# overlay), but wiped when we land on a different node. NFS can't hold podman
# graphroot (no xattrs), but it can hold compressed tarballs from `podman save`.
#
# Streams `podman save <ref> | zstd -T0 -3` to avoid an intermediate uncompressed
# tar and to shrink NFS footprint (~2 GB/image raw → ~0.5-1 GB/image compressed).
# On load, streams `zstd -dc <file> | podman load`.
#
# Usage:
#   bash sh/archive_images.sh save                      # snapshot bases + wrappers + builder (default)
#   bash sh/archive_images.sh save --wrappers-only      # RECOMMENDED: wrappers + builder only
#                                                       # (wrappers tars already contain base layers,
#                                                       #  so saving bases separately is ~500GB of waste;
#                                                       #  on load, re-pull base tags via ghcr — fast
#                                                       #  because layers are already local)
#   bash sh/archive_images.sh load              # restore all NFS archives into local podman
#   bash sh/archive_images.sh list              # show which archives exist + sizes
#   bash sh/archive_images.sh save --all        # snapshot every local image (not just swebench)
#   bash sh/archive_images.sh save --parallel 4
#
# Options:
#   --archive-dir PATH   override default ${MZ_CACHE}/podman-archive
#   --parallel N         parallel save/load workers (default 2)
#   --force              re-save even if archive exists for the same image id
#   --wrappers-only      skip base images (saves ~500 GB; bases re-pull from ghcr on next pod)
#   --zstd-level N       zstd level 1-19 (default 3 — fast, ~30% reduction)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
export PODMAN_IGNORE_CGROUPSV1_WARNING=1

ARCHIVE_DIR="${MZ_CACHE}/podman-archive"
PARALLEL=2
FORCE=false
SCOPE="sweb-only"   # sweb-only | all | wrappers-only
ZSTD_LEVEL=3

MODE="${1:-}"; shift || true

while (($#)); do
    case "$1" in
        --archive-dir) ARCHIVE_DIR="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        --force) FORCE=true; shift ;;
        --all) SCOPE="all"; shift ;;
        --wrappers-only) SCOPE="wrappers-only"; shift ;;
        --zstd-level) ZSTD_LEVEL="$2"; shift 2 ;;
        -h|--help) sed -n '4,34p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v zstd >/dev/null 2>&1 || { echo "zstd is required (sudo yum install zstd)" >&2; exit 2; }

mkdir -p "$ARCHIVE_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# Canonical filename mapping: repo:tag  ->  ${ARCHIVE_DIR}/<safe>.tar.zst
safe_name() {
    # docker.io/swebench/sweb.eval.x86_64.django_1776_django-12284:latest -> docker.io__swebench__sweb.eval.x86_64.django_1776_django-12284__latest
    echo "$1" | sed 's|[/:]|__|g'
}

sweb_image_list() {
    podman images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep -E 'sweb\.eval\.|localhost/swerex-builder|localhost/sweb-wrapper-' || true
}

wrappers_image_list() {
    # Wrapper tars already include base layers transitively, so we skip bases
    # here. On load, re-tag bases via `pull_bases_via_ghcr` — layers are local,
    # so podman just downloads the manifest and creates the tag (near-instant).
    podman images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null \
        | grep -E 'localhost/swerex-builder|localhost/sweb-wrapper-' || true
}

all_image_list() {
    podman images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -v '<none>:<none>' || true
}

# ── SAVE ────────────────────────────────────────────────────────────────────
# Streams podman save -> zstd to avoid a ~2 GB uncompressed intermediate tar
# per image. Set pipefail so a podman save failure isn't masked by a zstd OK.
save_one() {
    local ref="$1"
    local out="$ARCHIVE_DIR/$(safe_name "$ref").tar.zst"
    if ! $FORCE && [[ -f "$out" ]]; then
        echo "[skip] $ref (archive exists: $(du -h "$out" | awk '{print $1}'))"
        return 0
    fi
    local tmp="${out}.tmp.$$"
    if (set -o pipefail; podman save "$ref" 2>/dev/null | zstd -T0 -"$ZSTD_LEVEL" -q -o "$tmp"); then
        mv -f "$tmp" "$out"
        echo "[ok]   $ref -> $(basename "$out") ($(du -h "$out" | awk '{print $1}'))"
    else
        rm -f "$tmp"
        echo "[FAIL] $ref"
    fi
}
export -f save_one safe_name
export ARCHIVE_DIR FORCE ZSTD_LEVEL

# ── LOAD ────────────────────────────────────────────────────────────────────
load_one() {
    local tar="$1"
    local name; name=$(basename "$tar" .tar.zst)
    name=$(basename "$name" .tar)  # tolerate legacy uncompressed archives
    # derive original ref (reverse of safe_name)
    local ref; ref=$(echo "$name" | sed 's|__|/|' | sed 's|__|/|' | sed 's|__|:|')
    if ! $FORCE && podman image exists "$ref" 2>/dev/null; then
        echo "[skip] $ref (already loaded)"
        return 0
    fi
    local ok=false
    if [[ "$tar" == *.tar.zst ]]; then
        if (set -o pipefail; zstd -dc -T0 "$tar" 2>/dev/null | podman load >/dev/null 2>&1); then
            ok=true
        fi
    else
        podman load -i "$tar" >/dev/null 2>&1 && ok=true
    fi
    $ok && echo "[ok]   $ref" || echo "[FAIL] $ref (tar: $tar)"
}
export -f load_one
export FORCE

# ── Dispatch ────────────────────────────────────────────────────────────────
case "$MODE" in
    save)
        case "$SCOPE" in
            all)            IMGS=$(all_image_list) ;;
            wrappers-only)  IMGS=$(wrappers_image_list) ;;
            *)              IMGS=$(sweb_image_list) ;;
        esac
        N=$(echo "$IMGS" | grep -c . || true)
        (( N > 0 )) || { log "nothing to save."; exit 0; }
        log "Saving $N image(s) to $ARCHIVE_DIR with $PARALLEL parallel worker(s) (scope=$SCOPE, zstd=-$ZSTD_LEVEL)..."
        echo "$IMGS" | xargs -I{} -P "$PARALLEL" bash -c 'save_one "$@"' _ {}
        log "Save complete. Archive size:"
        du -sh "$ARCHIVE_DIR"
        ;;
    load)
        mapfile -t TARS < <(find "$ARCHIVE_DIR" -maxdepth 1 \( -name '*.tar.zst' -o -name '*.tar' \) -type f)
        (( ${#TARS[@]} > 0 )) || { log "no *.tar{,.zst} under $ARCHIVE_DIR."; exit 0; }
        log "Loading ${#TARS[@]} archive(s) with $PARALLEL parallel worker(s)..."
        printf '%s\n' "${TARS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'load_one "$@"' _ {}
        log "Load complete."
        ;;
    list)
        echo
        echo "Archive dir: $ARCHIVE_DIR"
        if [[ ! -d "$ARCHIVE_DIR" ]]; then echo "(missing)"; exit 0; fi
        total=0
        count=0
        while IFS= read -r f; do
            sz=$(stat -c%s "$f")
            total=$(( total + sz ))
            count=$(( count + 1 ))
            printf "  %10s  %s\n" "$(numfmt --to=iec --suffix=B "$sz")" "$(basename "$f")"
        done < <(find "$ARCHIVE_DIR" -maxdepth 1 \( -name '*.tar.zst' -o -name '*.tar' \) -type f | sort)
        echo
        echo "  ${count} archive(s), total $(numfmt --to=iec --suffix=B "$total")"
        ;;
    ""|-h|--help)
        sed -n '4,23p' "$0"; exit 0
        ;;
    *)
        echo "Unknown mode: $MODE (expected: save | load | list)" >&2
        exit 2
        ;;
esac
