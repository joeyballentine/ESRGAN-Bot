#!/bin/bash
shopt -s extglob

# Resizes LR 4x with Point filter and saves in ${TMP_FOLDER} folder
# Adds HR to ${TMP_FOLDER} folder
resizeLR() {
    FILENAME=$(basename -- "$file")
    FL="${FILENAME%.*}"
        firstRes1="`magick identify -format '%w' ${IMAGE_FIRST}`"
        secondRes1="`magick identify -format '%w' ${IMAGE_SECOND}`"
        oneEX=$((firstRes1*1))
        fourEX=$((firstRes1*4))
        eightEX=$((firstRes1*8))
        if [[ $secondRes1 = $oneEX ]]; then
            autoScale2="100%"
        elif [[ $secondRes1 = $fourEX ]]; then
            autoScale2="400%"
        elif [[ $secondRes1 = $eightEX ]]; then
            autoScale2="800%"
        else
            autoScale2="400%"
        fi
    firstFixed=$(basename ${IMAGE_FIRST})
    secondFixed=$(basename ${IMAGE_SECOND})
    ${MAGICK_COMMAND} convert ${IMAGE_FIRST} -filter point -resize ${autoScale2} ${TMP_FOLDER}/${firstFixed}
    ${MAGICK_COMMAND} convert ${IMAGE_SECOND} -filter point -resize ${UPSCALE_FACTOR_SECOND_IMAGE} ${TMP_FOLDER}/${secondFixed}
}

# Creates circle gradient with same dimensions as final montage
# Creates checkerboard pattern with same dimensions as final montage
# Composites Gradient and Checkerboard together
# Creates Two text Images with transparency and blurred shadows
# Adds 64px border to bottom of final montage
# Composites Montage over Background, then composites Text over Montage
mont() {
    ${MAGICK_COMMAND} montage -background black ${TMP_FOLDER}/${firstFixed} ${TMP_FOLDER}/${secondFixed} -tile ${TILE_DIM} -geometry +0+0 -depth 8 -define png:color-type=2 -depth 8 ${TMP_FOLDER}/montage.png
    ${MAGICK_COMMAND} mogrify -filter point -resize ${UPSCALE_FACTOR_GLOBAL} -bordercolor None -border 0x64 -gravity North -chop 0x64 -depth 8 ${TMP_FOLDER}/montage.png
    RES1="`${MAGICK_COMMAND} identify -format '%wx%h' ${TMP_FOLDER}/montage.png`"
    RES2="`${MAGICK_COMMAND} identify -format '%w' ${TMP_FOLDER}/montage.png`"
    ${MAGICK_COMMAND} convert -size ${RES1} pattern:checkerboard -depth 8 ${TMP_FOLDER}/tiles.png
    ${MAGICK_COMMAND} convert -size ${RES1} "radial-gradient:${COLOR_FIRST}-${COLOR_SECOND}" -depth 8 ${TMP_FOLDER}/gradient.png
    ${MAGICK_COMMAND} convert ${TMP_FOLDER}/gradient.png \( ${TMP_FOLDER}/tiles.png -alpha set -channel Alpha -evaluate set 40% \) -compose Overlay -composite -depth 8 ${TMP_FOLDER}/BG.png
    ${MAGICK_COMMAND} convert ${TMP_FOLDER}/BG.png ${TMP_FOLDER}/montage.png -composite -depth 8 ${TMP_FOLDER}/Montage_BG_temp1.png

    ${MAGICK_COMMAND} convert -channel RGBA -size ${RES1} -resize 200%x400% xc:none -gravity south -pointsize 144 -font "${FONT}" \
    -fill black -annotate -0+18 ${TEXT_FIRST} -blur 0x11 +noise Poisson -fill white \
    -stroke black -strokewidth 7 -annotate -0+32 ${TEXT_FIRST} -resize 25% -depth 8 ${TMP_FOLDER}/Montage_BG_temp2.png

    ${MAGICK_COMMAND} convert -channel RGBA -size ${RES1} -resize 200%x400% xc:none -gravity south -pointsize 144 -font "${FONT}" \
    -fill black -annotate +0+18 ${TEXT_SECOND} -blur 0x11 +noise Poisson -fill white \
    -stroke black -strokewidth 7 -annotate +0+32 ${TEXT_SECOND} -resize 25% -depth 8 ${TMP_FOLDER}/Montage_BG_temp3.png

    ${MAGICK_COMMAND} montage -channel RGBA -background none ${TMP_FOLDER}/Montage_BG_temp2.png ${TMP_FOLDER}/Montage_BG_temp3.png -tile 2x1 -geometry +0+0 -depth 8 ${TMP_FOLDER}/Montage_BG.png

    ${MAGICK_COMMAND} convert ${TMP_FOLDER}/Montage_BG_temp1.png ${TMP_FOLDER}/Montage_BG.png -composite -depth 8 ${OUT_FOLDER}/${IMAGE_OUTPUT}
}

# firstRes1="`magick identify -format '%w' ${IMAGE_FIRST}`"
# secondRes1="`magick identify -format '%w' ${IMAGE_SECOND}`"
# oneEX=$((firstRes1*1))
# fourEX=$((firstRes1*4))
# eightEX=$((firstRes1*8))
# if [[ $secondRes1 = $oneEX ]]; then
    # autoScale2="100%"
# elif [[ $secondRes1 = $fourEX ]]; then
    # autoScale2="400%"
# elif [[ $secondRes1 = $eightEX ]]; then
    # autoScale2="800%"
# fi

# Initialize Variables
COLOR_FIRST="rgb(125, 65, 130)"
COLOR_SECOND="rgb(255, 209, 65)"
IMAGE_FIRST="first.png"
IMAGE_SECOND="second.png"
IMAGE_OUTPUT="comparisonOutput.png"
TEXT_FIRST="LR"
TEXT_SECOND="HR"
TILE_DIM="2x1"
FONT="Rubik-Bold"
TMP_FOLDER=".temp1"
MAGICK_COMMAND="magick"
UPSCALE_FACTOR_FIRST_IMAGE="100%"
UPSCALE_FACTOR_SECOND_IMAGE="100%"
UPSCALE_FACTOR_GLOBAL="100%"
OUT_FOLDER="montages"

# argphase, fill variables with values from the user if specified  
# put arguments in quotations eg: -if="whateverInputImage.png" and -is="whateverSecondInputImage.png"  
for OPTION in "$@"; do
    case ${OPTION} in
        -if=*|--image-first=*)
            IMAGE_FIRST="${OPTION#*=}"
            shift
        ;;
        -is=*|--image-second=*)
            IMAGE_SECOND="${OPTION#*=}"
            shift
        ;;
        -io=*|--image-output=*)
            IMAGE_OUTPUT="${OPTION#*=}"
            shift
        ;;
        -cf=*|--color-first=*)
            COLOR_FIRST="${OPTION#*=}"
            shift
        ;;
        -cs=*|--color-second=*)
            COLOR_SECOND="${OPTION#*=}"
            shift
        ;;
        -tf=*|--text-first=*)
            TEXT_FIRST="${OPTION#*=}"
            shift
        ;;
        -ts=*|--text-second=*)
            TEXT_SECOND="${OPTION#*=}"
            shift
        ;;
        -td=*|--tile_dim=*)
            TILE_DIM="${OPTION#*=}"
            shift
        ;;
        -f=*|--font=*)
            FONT="${OPTION#*=}"
            shift
        ;;
        -tf=*|--tmp-folder=*)
            TMP_FOLDER="${OPTION#*=}"
            shift
        ;;
        -m=*|--magick-command=*)
            MAGICK_COMMAND="${OPTION#*=}"
            shift
        ;;
        -uf=*|--rez-1=*)
            UPSCALE_FACTOR_FIRST_IMAGE="${OPTION#*=}"
            shift
        ;;
        -us=*|--rez-2=*)
            UPSCALE_FACTOR_SECOND_IMAGE="${OPTION#*=}"
            shift
        ;;
        -ug=*|--Lres_1=*)
            UPSCALE_FACTOR_GLOBAL="${OPTION#*=}"
            shift
        ;;
        -of=*|--out_folder=*)
            OUT_FOLDER="${OPTION#*=}"
            shift
        ;;
        # Any other input
        *)
            echo "usage: $@ ..."
            echo "-if=, --image-first=\"<path to first image>\" (default: ${IMAGE_FIRST})"
            echo "-is=, --image-second=\"<path to second image>\" (default: ${IMAGE_SECOND})"
            echo "-io=, --image-output=\"<path to output image>\" (default: ${IMAGE_OUTPUT})"
            exit 1
        ;;
    esac
done

# Create a tmp folder
mkdir -p ${TMP_FOLDER}
mkdir -p ${OUT_FOLDER}

# work
resizeLR
mont

# clean up
rm -rf ${TMP_FOLDER}
