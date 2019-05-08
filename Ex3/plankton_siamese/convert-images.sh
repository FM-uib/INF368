
# safety first
set -e -o pipefail
shopt -s failglob 

# resize to 299x299, but keep image size
# drop -resize if zooming with crop is better (but maybe not?)

BASE=Euc  #ZooScanSet/imgs
OUT=data

ls "$BASE" | tail -n +1 | while read dir; do #org tail +14
   echo "$dir"
   mkdir -p "$OUT/$dir"
   ls "$BASE/$dir" | while read f; do 
          convert -resize 299x299 "$BASE/$dir/$f" -background white -gravity center -extent 299x299 "$OUT/$dir/$f"
   done
done
