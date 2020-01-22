for file in $1/*.*; do
    FILENAME=$(basename -- "$file")
    FL="${FILENAME%.*}"
    magick convert $file -bordercolor Black -border 8x8 $1/${FL}_border.png
    rm -f $file
    mv $1/${FL}_border.png $1/${FL}.png
    magick convert $1/${FL}.png -crop 3x3+16+16@ +repage +adjoin $1/${FL}_%02d.png
    rm -f $1/${FL}.png
done