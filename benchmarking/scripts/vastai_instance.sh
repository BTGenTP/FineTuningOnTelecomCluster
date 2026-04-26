#!/usr/bin/env bash
# ============================================================================
# vastai_instance.sh — SSH et cycle de vie des instances vast.ai
# ============================================================================
# Prérequis : CLI `vastai` (pip install vastai), `vastai set api-key …`
#
# Variables optionnelles :
#   VASTAI_SSH_KEY   chemin clé privée (défaut : ~/.ssh/id_rsa)
#   VASTAI_SSH_PUB   chemin clé publique (défaut : ${VASTAI_SSH_KEY}.pub)
#
# Usage :
#   ./scripts/vastai_instance.sh list
#   ./scripts/vastai_instance.sh connect <instance_id> [-- options_ssh…]
#   ./scripts/vastai_instance.sh url <instance_id>
#   ./scripts/vastai_instance.sh attach-key <instance_id>
#   ./scripts/vastai_instance.sh destroy <instance_id>
#   ./scripts/vastai_instance.sh destroy-all [--yes]
#   ./scripts/vastai_instance.sh start|stop|reboot <instance_id>
#   ./scripts/vastai_instance.sh logs <instance_id> [--tail N]
#   ./scripts/vastai_instance.sh scp-url <instance_id>
#   ./scripts/vastai_instance.sh copy <id>:/chemin/remote ./chemin/local
#   ./scripts/vastai_instance.sh bench …   → délègue à vastai_run.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VASTAI_RUN="${SCRIPT_DIR}/vastai_run.sh"

SSH_KEY="${VASTAI_SSH_KEY:-$HOME/.ssh/id_rsa}"
SSH_PUB="${VASTAI_SSH_PUB:-${SSH_KEY}.pub}"

die() { echo "ERROR: $*" >&2; exit 1; }

need_vastai() {
    command -v vastai &>/dev/null || die "vastai introuvable. pip install vastai && vastai set api-key …"
}

usage() {
    cat <<'EOF'
vastai_instance.sh — SSH et cycle de vie des instances vast.ai

Variables : VASTAI_SSH_KEY, VASTAI_SSH_PUB (défaut ~/.ssh/id_rsa)

Commands:
  list [-q|-a]              vastai show instances (-q = IDs only, -a = all pages)
  url <id>                  affiche l’URL ssh://…
  attach-key <id>           vastai attach ssh (clé publique)
  connect <id> [-- …]      attach-key + ssh (tout après -- est passé à ssh)
  destroy <id>            détruit une instance (-y)
  destroy-all [--yes]     détruit toutes les instances (confirmation sauf --yes)
  start|stop|reboot <id>   cycle de vie instance
  logs <id> [args…]       vastai logs (ex. --tail 200)
  scp-url <id>            vastai scp-url (aide transfert fichiers)
  copy <src> <dst>        vastai copy -i clé (ex. ID:/workspace/f.tgz ./runs/vastai/)
  bench [args…]           lance ./scripts/vastai_run.sh (nouveau benchmark)
EOF
    exit 0
}

cmd_list() {
    need_vastai
    vastai show instances "$@"
}

cmd_url() {
    local id="${1:?usage: $0 url <instance_id>}"
    need_vastai
    vastai ssh-url "$id"
}

cmd_attach_key() {
    local id="${1:?usage: $0 attach-key <instance_id>}"
    [[ -f "$SSH_PUB" ]] || die "Clé publique absente : $SSH_PUB (export VASTAI_SSH_PUB=…)"
    [[ -f "$SSH_KEY" ]] || die "Clé privée absente : $SSH_KEY"
    need_vastai
    vastai attach ssh "$id" "$SSH_PUB"
}

cmd_connect() {
    local id="${1:?usage: $0 connect <instance_id> [-- …ssh args…]}"
    shift || true
    if [[ "${1:-}" == "--" ]]; then
        shift
    fi
    [[ -f "$SSH_KEY" ]] || die "Clé privée absente : $SSH_KEY"
    need_vastai
    if [[ -f "$SSH_PUB" ]]; then
        vastai attach ssh "$id" "$SSH_PUB" || echo "[warn] attach ssh a échoué ou clé déjà présente — poursuite…" >&2
    else
        echo "[warn] Pas de $SSH_PUB — connexion sans nouvel attach." >&2
    fi
    local url
    url=$(vastai ssh-url "$id")
    echo "# ssh -i ${SSH_KEY@Q} ${url@Q}" >&2
    exec ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "$url" "$@"
}

cmd_destroy() {
    local id="${1:?usage: $0 destroy <instance_id>}"
    need_vastai
    vastai destroy instance -y "$id"
    echo "Instance $id détruite."
}

cmd_destroy_all() {
    need_vastai
    local yes=0
    for a in "$@"; do
        [[ "$a" == "--yes" || "$a" == "-y" ]] && yes=1
    done
    if [[ "$yes" != 1 ]]; then
        read -r -p "Détruire TOUTES les instances du compte ? [y/N] " ans || true
        [[ "${ans:-}" =~ ^[yY] ]] || die "annulé."
    fi
    mapfile -t ids < <(vastai show instances -q -a 2>/dev/null || true)
    if [[ "${#ids[@]}" -eq 0 ]]; then
        echo "Aucune instance à détruire."
        exit 0
    fi
    echo "IDs : ${ids[*]}"
    vastai destroy instances -y "${ids[@]}"
    echo "Terminé."
}

cmd_lifecycle() {
    local verb="$1"
    local id="${2:?usage: $0 ${verb} <instance_id>}"
    need_vastai
    vastai "${verb}" instance "$id"
}

cmd_logs() {
    local id="${1:?usage: $0 logs <instance_id> [vastai logs options…]}"
    shift || true
    need_vastai
    vastai logs "$id" "$@"
}

cmd_scp_url() {
    local id="${1:?usage: $0 scp-url <instance_id>}"
    need_vastai
    vastai scp-url "$id"
}

cmd_copy() {
    # `vastai copy` often breaks with rsync "@ERROR: Unknown module" (vast-cli#326) — use scp + API metadata.
    [[ $# -eq 2 ]] || die "usage: $0 copy <instance_id>:/chemin/fichier_remote /chemin/fichier_local"
    [[ "$1" == *:* ]] || die "format attendu: ID:/chemin (ex. 35393590:/workspace/results_llama.tar.gz)"
    [[ -f "$SSH_KEY" ]] || die "Clé privée absente : $SSH_KEY"
    need_vastai

    local spec="$1" dst="$2"
    local id="${spec%%:*}"
    local rpath="${spec#*:}"
    [[ -n "$id" && -n "$rpath" ]] || die "chemin remote invalide"

    local meta host port
    meta=$(vastai show instance "$id" --raw 2>/dev/null) || die "vastai show instance $id a échoué"
    read -r host port < <(echo "$meta" | python3 -c '
import json, sys
d = json.load(sys.stdin)
ip = (d.get("public_ipaddr") or "").strip()
md = d.get("machine_dir_ssh_port")
sh = (d.get("ssh_host") or "").strip()
sp = d.get("ssh_port")

def ok_port(p):
    return isinstance(p, int) and 1 <= p < 65535

# Préférer la passerelle vast (ssh_host:ssh_port) : c’est là que « attach ssh » installe ta clé.
# public_ipaddr:machine_dir_ssh_port peut exiger un mot de passe root si la clé n’y est pas poussée.
if sh and ok_port(sp):
    print(sh, sp)
elif ip and ok_port(md):
    print(ip, md)
else:
    sys.exit(1)
') || die "Impossible de déduire hôte/port SSH (voir vastai show instance $id --raw)"

    echo "# scp (passerelle vast, clé attachée) — P $port root@$host:$rpath → $dst" >&2
    scp -i "$SSH_KEY" -P "$port" -o StrictHostKeyChecking=accept-new \
        -r "root@${host}:${rpath}" "$dst"
}

cmd_bench() {
    [[ -x "$VASTAI_RUN" ]] || die "Script introuvable : $VASTAI_RUN"
    exec "$VASTAI_RUN" "$@"
}

main() {
    [[ $# -ge 1 ]] || usage
    local cmd="$1"
    shift

    case "$cmd" in
        -h|--help|help) usage ;;
        list|ls)        cmd_list "$@" ;;
        url)            cmd_url "$@" ;;
        attach-key|attach)
                        cmd_attach_key "$@" ;;
        connect|ssh)    cmd_connect "$@" ;;
        destroy)        cmd_destroy "$@" ;;
        destroy-all|destroyall)
                        cmd_destroy_all "$@" ;;
        start)          cmd_lifecycle start "$@" ;;
        stop)           cmd_lifecycle stop "$@" ;;
        reboot)         cmd_lifecycle reboot "$@" ;;
        logs)           cmd_logs "$@" ;;
        scp-url)        cmd_scp_url "$@" ;;
        copy)           cmd_copy "$@" ;;
        bench)          cmd_bench "$@" ;;
        *)              die "commande inconnue : $cmd (voir $0 --help)" ;;
    esac
}

main "$@"
