const imagemagickCli = require('imagemagick-cli');
const fs = require(`fs`);
const sizeOf = require('image-size');
const util = require('util');
const exec = util.promisify(require('child_process').exec);

exports.downscale = async (imagePath, amount, filter) => {
    let size = (1.0 / amount) * 100.0 + '%';
    return new Promise((resolve, reject) => {
        exec(
            `magick mogrify -resize ${size} -filter ${filter} -format png ${imagePath.replace(
                '\\',
                '/'
            )}`
        )
            .then((stdout, stderr) => {
                if (stderr) {
                    console.error(`error: ${stderr}`);
                    reject();
                } else {
                    resolve();
                }
            })
            .catch(reject());
    });
};

exports.convertToPNG = async (imagePath) => {
    return new Promise((resolve, reject) => {
        if (
            imagePath
                .split('.')
                .pop()
                .toLowerCase() === 'png'
        ) {
            resolve(imagePath);
        } else {
            exec(`magick mogrify -format png ${imagePath}`)
                .then((stdout, stderr) => {
                    if (stderr) {
                        console.error(`error: ${stderr}`);
                        reject();
                    } else {
                        fs.unlink(imagePath, (err) => {
                            if (err) {
                                console.error(err);
                                reject();
                            }
                            resolve(stdout ? stdout : stderr);
                        });
                    }
                })
                .catch(reject());
        }
    });
};

exports.split = async (imagePath) => {
    return new Promise((resolve, reject) => {
        imagemagickCli
            .exec(`magick mogrify -bordercolor Black -border 8x8 ${imagePath}`)
            .then(({ stdout, stderr }) => {
                if (stderr) {
                    console.log(stderr);
                    reject();
                }
                imagemagickCli
                    .exec(
                        `magick mogrify -crop 3x3+16+16@ +repage ${imagePath}`
                    )
                    .then(({ stdout, stderr }) => {
                        if (stderr) {
                            console.log(stderr);
                            reject();
                        }
                        resolve(stdout ? stdout : stderr);
                    });
            });
    });
};

exports.merge = async (resultDir, imageName, lrDir) => {
    let lrSize;
    let resultSize;
    for (file of fs.readdirSync(lrDir)) {
        lrSize = sizeOf(lrDir + '/' + file);
        console.log(lrSize);
        break;
    }
    for (file of fs.readdirSync(resultDir)) {
        resultSize = sizeOf(resultDir + '/' + file);
        console.log(resultSize);
        break;
    }
    let scale = Math.sqrt(
        (resultSize.width * resultSize.height) / (lrSize.width * lrSize.height)
    );
    let overlap = 8 * scale;
    return new Promise((resolve, reject) => {
        imagemagickCli
            .exec(
                `magick mogrify -alpha set -virtual-pixel transparent -channel A -blur 0x${scale} -level 50%,100% +channel ${resultDir}/*.png`
            )
            .then(({ stdout, stderr }) => {
                if (stderr) {
                    console.log(stderr);
                    reject();
                }
                imagemagickCli
                    .exec(
                        `magick montage ${resultDir}/*.png -geometry -${overlap}-${overlap} -background black -depth 8 -define png:color-type=2 ${resultDir}/${imageName}_rlt.png`
                    )
                    .then(({ stdout, stderr }) => {
                        if (stderr) {
                            console.log(stderr);
                            reject();
                        }
                        fs.readdir(resultDir, (err, files) => {
                            if (err) {
                                console.log(err);
                            }

                            for (file of files) {
                                if (file !== `${imageName}_rlt.png`) {
                                    fs.unlinkSync(resultDir + '/' + file);
                                }
                            }

                            resolve(stdout ? stdout : stderr);
                        });
                    });
            });
    });
};
