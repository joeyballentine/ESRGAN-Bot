for file in $1/*.*; do
    image1=$file
    break 1
done

for file in $2/*.*; do
    image2=$file
    break 1
done

firstRes="`magick identify -format '%w' ${image1}`"
secondRes="`magick identify -format '%w' ${image2}`"
echo ${firstRes}
echo ${secondRes}
scale=$((secondRes / firstRes))
echo ${scale}

overlap=$((8 * scale))
echo ${overlap}

magick mogrify -alpha set -virtual-pixel transparent -channel A -blur 0x4 -level 50%,100% +channel $2/*.png
magick montage $2/*.png -geometry -${overlap}-${overlap} -background black -depth 8 -define png:color-type=2 $2/result.png