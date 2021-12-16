#!/bin/bash

echo “Processing…”
wget -q -U "Mozilla/5.0" –post-file file.flac –header "Content-Type: audio/x-flac; rate=16000" -O – "https://www.google.com/speech-api/v2/recognize?output=json&lang=en-us&key=<your key>" | cut -d '' -f12 >stt.txt

echo -n "You Said: "
cat stt.txt

rm file.flac > /dev/null 2>&1