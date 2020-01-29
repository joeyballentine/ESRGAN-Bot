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
    ${MAGICK_COMMAND} convert ${IMAGE_FIRST} -filter point -resize ${autoScale2} ${TMP_FOLDER}/${firstFixed} &&
    ${MAGICK_COMMAND} convert ${IMAGE_SECOND} -filter point -resize ${UPSCALE_FACTOR_SECOND_IMAGE} ${TMP_FOLDER}/${secondFixed}
}

# Creates circle gradient with same dimensions as final montage
# Creates checkerboard pattern with same dimensions as final montage
# Composites Gradient and Checkerboard together
# Creates Two text Images with transparency and blurred shadows
# Adds 64px border to bottom of final montage
# Composites Montage over Background, then composites Text over Montage
mont() {
    randName1="`cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 8 | head -n1`"
    FILENAME=$(basename -- "$file")
    FL="${FILENAME%.*}"
    BG1="${randName1}_BG1.png"
    tmpMont="${randName1}_montage.png"
    tmpMont1="${randName1}_montage_BG1.png"
    tmpMont2="${randName1}_montage_BG2.png"
    tmpMont3="${randName1}_montage_BG3.png"
    tmpMontFin="${randName1}_montage_BGfin.png"
    tmpTiles="${randName1}_tiles.png"
    tmpGradient="${randName1}_gradient.png"

    
    ${MAGICK_COMMAND} montage -background black ${TMP_FOLDER}/${firstFixed} ${TMP_FOLDER}/${secondFixed} -tile ${TILE_DIM} -geometry +0+0 -depth 8 -define png:color-type=2 -depth 8 ${TMP_FOLDER}/${tmpMont} &&
    ${MAGICK_COMMAND} mogrify -filter point -resize ${UPSCALE_FACTOR_GLOBAL} -bordercolor None -border 0x64 -gravity North -chop 0x64 -depth 8 ${TMP_FOLDER}/${tmpMont} &&
    RES1="`${MAGICK_COMMAND} identify -format '%wx%h' ${TMP_FOLDER}/${tmpMont}`" &&
    RES2="`${MAGICK_COMMAND} identify -format '%w' ${TMP_FOLDER}/${tmpMont}`" &&
    RES3="`${MAGICK_COMMAND} identify -format '%h' ${TMP_FOLDER}/${tmpMont}`" &&
    RGRw=$((RES2/4))
    RGRh=$((RES3/4))
    ResX4="${RGRw}x${RGRh}"
    ${MAGICK_COMMAND} convert -size ${ResX4} xc: -tile ./scripts/checkerboard6x6.png -draw "color 0,0 reset" -filter Point -resize ${RES1} -depth 8 ${TMP_FOLDER}/${tmpTiles} &&
    ${MAGICK_COMMAND} convert -size ${ResX4} "radial-gradient:${COLOR_FIRST}-${COLOR_SECOND}" -filter Spline -resize ${RES1} -depth 8 ${TMP_FOLDER}/${tmpGradient} &&
    ${MAGICK_COMMAND} convert ${TMP_FOLDER}/${tmpGradient} \( ${TMP_FOLDER}/${tmpTiles} -alpha set -channel Alpha -evaluate set 40% \) -compose Overlay -composite -depth 8 ${TMP_FOLDER}/${BG1} &&
    ${MAGICK_COMMAND} convert ${TMP_FOLDER}/${BG1} ${TMP_FOLDER}/${tmpMont} -composite -depth 8 ${TMP_FOLDER}/${tmpMont1} &&

    ${MAGICK_COMMAND} convert -channel RGBA -size ${RES1} -resize 100%x200% xc:none -gravity south -pointsize 75 -font "${FONT}" \
    -fill white -stroke black -strokewidth 4 -annotate -0+17 ${TEXT_FIRST} -resize 50% -depth 8 ${TMP_FOLDER}/${tmpMont2} &&

    ${MAGICK_COMMAND} convert -channel RGBA -size ${RES1} -resize 100%x200% xc:none -gravity south -pointsize 75 -font "${FONT}" \
    -fill white -stroke black -strokewidth 4 -annotate +0+17 ${TEXT_SECOND} -resize 50% -depth 8 ${TMP_FOLDER}/${tmpMont3} &&

    ${MAGICK_COMMAND} montage -channel RGBA -background none ${TMP_FOLDER}/${tmpMont2} ${TMP_FOLDER}/${tmpMont3} -tile 2x1 -geometry +0+0 -depth 8 ${TMP_FOLDER}/${tmpMontFin} &&

    ${MAGICK_COMMAND} convert ${TMP_FOLDER}/${tmpMont1} ${TMP_FOLDER}/${tmpMontFin} -composite -depth 8 ${OUT_FOLDER}/${IMAGE_OUTPUT}
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
resizeLR &&
mont

### clean up
rm -rf ${TMP_FOLDER}
