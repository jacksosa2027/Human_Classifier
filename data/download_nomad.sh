#!/usr/bin/env bash
# Downloads and converts all 100 NOMAD actors (images + labels), one actor at
# a time, streaming straight into the YOLO train/val/test split.
#
# Each actor is downloaded, converted+resized into nomad_dataset/, then its
# raw copy is deleted before the next actor starts. This keeps peak disk
# usage to roughly "prepared output so far + one actor's raw" instead of
# needing the full ~250GB raw NOMAD dataset on disk at once — see
# data/prepare_dataset.py for the --single-actor / --max-size / --keep-raw
# flags this relies on.
#
# Setup (one-time):
#   1. Install rclone:       sudo apt install rclone
#   2. Configure a remote:   rclone config
#                            -> New remote -> name it "gdrive" -> type: Google Drive
#                            -> Leave client_id/secret blank (use rclone's defaults)
#                            -> scope: drive.readonly -> follow browser auth flow
#   3. Run this script:      bash data/download_nomad.sh
#
# The NOMAD folder is accessed directly by its Google Drive folder ID, so you
# don't need to locate it inside your own Drive — rclone reaches it through the
# shared link regardless of where it lives.

set -euo pipefail

REMOTE="gdrive"                                    # name you gave the remote in rclone config
FOLDER_ID="1zRiOzedR-PzO1bps5I1vb6jtVoQHFWzg"   # NOMAD Google Drive folder ID
DEST="./NOMAD"                                     # local destination for the transient raw actor being processed
OUT="./nomad_dataset"                              # final YOLO-format split (this is what training reads)
FIRST_ACTOR=1
LAST_ACTOR=100
ALTITUDES="10,30"                                  # aerial distances (meters) to keep — match your mission's flight altitude.
                                                    # Farther tiers (50/70/90) shrink people to a few pixels and are mostly
                                                    # undetectable at training resolution — see data/prepare_dataset.py --altitudes

RCLONE_FLAGS=(
    --drive-root-folder-id "$FOLDER_ID"
    --transfers 8          # parallel file transfers
    --progress
    --stats-one-line
)

mkdir -p "$DEST"

echo "Streaming actors $FIRST_ACTOR-$LAST_ACTOR from NOMAD (download -> convert -> delete raw, one at a time)..."

for i in $(seq -f "%03g" "$FIRST_ACTOR" "$LAST_ACTOR"); do
    actor="Actor${i}"
    echo ""
    echo "==> $actor (images)"
    rclone copy "${REMOTE}:images/${actor}" "${DEST}/images/${actor}" "${RCLONE_FLAGS[@]}"

    echo "==> $actor (labels)"
    rclone copy "${REMOTE}:labels/${actor}" "${DEST}/labels/${actor}" "${RCLONE_FLAGS[@]}"

    echo "==> $actor (convert + delete raw)"
    python data/prepare_dataset.py --root "$DEST" --out "$OUT" \
        --single-actor "$actor" --actor-range "$FIRST_ACTOR" "$LAST_ACTOR" --altitudes "$ALTITUDES"
done

echo ""
echo "Done. Prepared dataset saved to: $(realpath "$OUT")"
echo "Next: python data/validate_dataset.py --data $(realpath "$OUT")"
