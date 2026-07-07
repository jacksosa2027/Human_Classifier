#!/usr/bin/env bash
# Downloads the first 20 actors (images + labels) from the NOMAD Google Drive.
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
DEST="./NOMAD"                                     # local destination
FIRST_ACTOR=1
LAST_ACTOR=95

RCLONE_FLAGS=(
    --drive-root-folder-id "$FOLDER_ID"
    --transfers 8          # parallel file transfers
    --progress
    --stats-one-line
)

mkdir -p "$DEST"

echo "Downloading actors $FIRST_ACTOR–$LAST_ACTOR from NOMAD..."

for i in $(seq -f "%03g" "$FIRST_ACTOR" "$LAST_ACTOR"); do
    actor="Actor${i}"
    echo ""
    echo "==> $actor (images)"
    rclone copy "${REMOTE}:images/${actor}" "${DEST}/images/${actor}" "${RCLONE_FLAGS[@]}"

    echo "==> $actor (labels)"
    rclone copy "${REMOTE}:labels/${actor}" "${DEST}/labels/${actor}" "${RCLONE_FLAGS[@]}"
done

echo ""
echo "Done. Dataset saved to: $(realpath "$DEST")"
echo "Next: python data/prepare_dataset.py --root $(realpath "$DEST")"
