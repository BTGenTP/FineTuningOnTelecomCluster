#!/bin/bash
# Auto-destroy Vast.ai instances when fine-tuning completes.
# Run on RPi5 in background: nohup bash vastai_auto_destroy.sh &

set -u
export PATH="$HOME/.local/bin:$PATH"

# Instance configs: ID SSH_HOST SSH_PORT LABEL
INSTANCES=(
    "34239383 ssh8.vast.ai 39382 qwen14b"
    "34239388 ssh2.vast.ai 39388 gemma9b"
)

CHECK_INTERVAL=300  # check every 5 minutes

log() { echo "[$(date '+%H:%M:%S')] $*"; }

for inst in "${INSTANCES[@]}"; do
    read -r ID HOST PORT LABEL <<< "$inst"
    log "Monitoring $LABEL (instance $ID) at $HOST:$PORT"
done

while [ ${#INSTANCES[@]} -gt 0 ]; do
    REMAINING=()
    for inst in "${INSTANCES[@]}"; do
        read -r ID HOST PORT LABEL <<< "$inst"

        # Check if python training is still running
        RUNNING=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
            -p "$PORT" "root@$HOST" \
            'pgrep -f "finetune_llama3_nav4rail" >/dev/null 2>&1 && echo YES || echo NO' \
            2>/dev/null)

        if [ "$RUNNING" = "NO" ]; then
            # Training finished — check if adapter was saved
            SAVED=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
                -p "$PORT" "root@$HOST" \
                'ls /workspace/output_*/lora_adapter/adapter_config.json 2>/dev/null && echo FOUND || echo MISSING' \
                2>/dev/null)

            if echo "$SAVED" | grep -q "FOUND"; then
                log "✓ $LABEL: training COMPLETE, adapter saved. Downloading..."

                # Create local dir and download adapter + logs
                DEST="$HOME/nav4rail_finetune_results/$LABEL"
                mkdir -p "$DEST"
                scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
                    -P "$PORT" -r "root@$HOST:/workspace/output_*/lora_adapter" "$DEST/" 2>&1
                scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
                    -P "$PORT" "root@$HOST:/workspace/train.log" "$DEST/train.log" 2>&1

                # Verify download
                if [ -f "$DEST/lora_adapter/adapter_config.json" ]; then
                    SIZE=$(du -sh "$DEST/lora_adapter" | cut -f1)
                    log "✓ $LABEL: adapter downloaded ($SIZE) to $DEST/lora_adapter"
                    log "✓ $LABEL: destroying instance $ID..."
                    vastai destroy instance "$ID" 2>&1
                    log "✓ $LABEL: instance $ID destroyed."
                else
                    log "⚠ $LABEL: download FAILED. NOT destroying instance $ID."
                    log "  Manual download: scp -P $PORT -r root@$HOST:/workspace/output_*/lora_adapter $DEST/"
                    REMAINING+=("$inst")
                fi
            else
                log "⚠ $LABEL: process stopped but no adapter found. Check logs! NOT destroying."
                log "  Manual check: ssh -p $PORT root@$HOST 'tail -20 /workspace/train.log'"
                REMAINING+=("$inst")
            fi
        else
            log "… $LABEL: still training"
            REMAINING+=("$inst")
        fi
    done
    INSTANCES=("${REMAINING[@]}")

    if [ ${#INSTANCES[@]} -gt 0 ]; then
        sleep "$CHECK_INTERVAL"
    fi
done

log "All instances destroyed. Exiting."
