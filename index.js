// TODO:
// Fix bug where upscaling a montage and montaging it send the results twice

// Discord.js import
const Discord = require('discord.js');
const client = new Discord.Client();

const request = require(`request`);
const {
    downloadModel,
    downloadImage
} = require('./download.js');
const {
    downscale,
    convertToPNG,
    split,
    merge
} = require('./imageUtils.js');

// File stuff
const fs = require(`fs`);
const fsExtra = require('fs-extra');
const sizeOf = require('image-size');
const isImageUrl = require('is-image-url');
const FuzzyMatching = require('fuzzy-matching');

// Shell stuff
const {
    PythonShell
} = require('python-shell');
const shell = require('shelljs');

// The image upscale queue
const queue = new Map();

// Configuration
const {
    token,
    prefix,
    pixelLimit,
    esrganPath
} = require('./config.json');

const models = new FuzzyMatching(fs.readdirSync(`${esrganPath}/models/`));

client.on('ready', () => {
    console.log(`Logged in as ${client.user.tag}!`);
    emptyDirs();
});

client.on('error', console.error);

client.on('message', async (message) => {
    if (!message.content.startsWith(prefix) || message.author.bot) return;
    const args = message.content.slice(prefix.length).split(' ');
    const command = args.shift().toLowerCase();

    if (command === 'upscale') {
        if (!args.length) {
            return message.channel.send(
                `You didn't provide any arguments, ${message.author}!`
            );
        }

        // Grabs the url of the image whether its an attachment or a url
        let url;
        if (message.attachments.first()) {
            url = message.attachments.first().url;
        } else if (isImageUrl(args[0])) {
            // Strips the url off of the args if a url is given
            url = args.shift();
        } else {
            // If no image is given the bot will error
            return message.channel.send('Not a valid command.');
        }

        // Sanitizing the url
        // url = url.split('&')[0];
        // url = url.split('?')[0];

        // Downloads the image
        let image = url.split('/').pop();

        // Gets the model name from the model argument
        let model = models.get(
            args[0].includes('.pth') ? args[0] : args[0] + '.pth'
        ).value;

        // Checks to make sure model name is valid (exists and is spelled right)
        if (!fs.readdirSync(esrganPath + '/models/').includes(model)) {
            return message.channel.send('Not a valid model.');
        }

        // The job sent for processing
        let upscaleJob = {
            url: url,
            model: model,
            image: image,
            downscale: false,
            filter: 'box',
            montage: false,
            message: message,
            split: false
        };

        // Parsing the extra arguments

        // Downscale
        if (args.includes('-downscale')) {
            upscaleJob.downscale = args[args.indexOf('-downscale') + 1];
        }

        // Filter
        if (args.includes('-filter')) {
            upscaleJob.filter = args[args.indexOf('-filter') + 1];
        }

        // Montage
        if (args.includes('-montage')) {
            upscaleJob.montage = true;
        }

        // Adds to the queue and starts upscaling if not already started.
        if (!queue.get(0)) {
            const queueConstruct = {
                textChannel: message.channel,
                jobs: []
            };
            queue.set(0, queueConstruct);
            queue.get(0).jobs.push(upscaleJob);

            try {
                message.channel.send(
                    `${upscaleJob.image} is being processed using ${upscaleJob.model}.`
                );
                await process(queue.get(0).jobs[0]);
            } catch (err) {
                // If something goes wrong here we just reset the entire queue
                // This probably isn't ideal but it's what the music bots do
                console.log(err);
                queue.delete(0);
                return message.channel.send(err);
            }
        } else {
            queue.get(0).jobs.push(upscaleJob);
            return message.channel.send(
                `${image} has been added to the queue! Your image is #${
                    queue.get(0).jobs.length
                } in line for processing.`
            );
        }
    } else if (command === 'models') {
        let files = fs.readdirSync(`${esrganPath}/models/`);
        let table = require('markdown-table');
        let models = [];
        for (let i = 0; i < files.length; i = i + 4) {
            models.push(files.slice(i, i + 4));
        }
        return message.channel.send(
            '```' +
            table(models, {
                rule: false
            }) +
            '```'
        );
    } else if (command === 'add') {
        let modelName = args[1].includes('.pth') ? args[1] : args[1] + '.pth';
        models.add(modelName);
        let url = args[0];
        if (url.includes('drive.google.com')) {
            url = url.replace('/view', '');
            url = url.split(
                `https://docs.google.com/uc?export=download&id=${url
                    .split('/')
                    .pop()}`
            );
        }
        message.channel.send('Adding model...');
        downloadModel(args[0], `${esrganPath}/models/`, modelName).then(() => {
            return message.channel.send(`Model ${modelName} successfully added.`);
        });
    } else if (command === 'help') {
        let help = `
Commands:

\`${prefix}upscale [model]\` // Upscales attached image using specified model

\`${prefix}upscale [url] [model]\` // Upscales linked image using specified model

\`${prefix}add [model url] [nickname]\` // Adds model from url, with a nickname (to avoid typing out long model names)

\`${prefix}models\` // Lists all models

\`${prefix}help\` // Shows this information again

Optional upscale args:

\`-downscale [amount]\` // Downscales the image by the amount listed

\`-filter [imagemagick filter]\` // Filter to be used for downscaling. Must be valid imagemagick filter. Defaults to box.

\`-montage\` // Creates aside by side comparison of the LR and result after upscaling

Example: \`${prefix}upscale www.imageurl.com/image.png 4xBox.pth -downscale 4 -filter point -montage\``;
        return message.channel.send(help);
    }
});

client.login(token);

// Empties LR and results folders
function emptyDirs() {
    fsExtra.emptyDirSync(esrganPath + '/results/');
    fsExtra.emptyDirSync(esrganPath + '/LR/');
}

// Processes an upscale job
async function process(job) {
    const parsePath = require('parse-filepath');

    let image = parsePath(await downloadImage(job.url, esrganPath + '/LR/'));
    console.log(image);

    if (!['.png', '.jpg', '.jpeg'].includes(image.ext.toLowerCase())) {
        return job.message.channel
            .send(
                `Sorry, ${job.message.author}, that image cannot be processed.`
            )
            .then(() => {
                processNext();
            });
    }

    if (image.ext.toLowerCase() !== '.png') {
        await convertToPNG(image.path).catch((error) => {
            console.log(error);
            return job.message.reply(
                'Sorry, there was an error processing your image. [c]'
            ).then(() => {
                processNext();
            });
        });
        image = parsePath(image.dir + '/' + image.name + '.png');
    }

    // Checks image to see if it should split
    let dimensions = sizeOf(image.path);
    if (job.downscale) {
        dimensions.width /= job.downscale;
        dimensions.height /= job.downscale;
    }
    if (dimensions.width >= pixelLimit || dimensions.height >= pixelLimit) {
        job.split = true;
    }

    // Downscales the image if argument provided
    if (job.downscale) {
        console.log('Downscaling...');
        await downscale(image.path, job.downscale, job.filter).catch(
            (error) => {
                console.log(error);
                return job.message.reply(
                    'Sorry, there was an error processing your image. [d]'
                ).then(() => {
                    processNext();
                });
            }
        );
    }

    // Splits if needed
    if (job.split) {
        console.log('Splitting...');
        await split(image.path).catch((error) => {
            console.log(error);
            return job.message.reply(
                'Sorry, there was an error processing your image. [s]'
            ).then(() => {
                processNext();
            });
        });
    }

    // Upscales the image(s)
    console.log('Upscaling...');
    await upscale(job.model).catch((error) => {
        console.log(error);
        return job.message.reply(
            'Sorry, there was an error processing your image. [u]'
        ).then(() => {
            processNext();
        });
    });

    // Merges the images if split was needed
    if (job.split) {
        console.log('Merging...');
        await merge(
            `${esrganPath}/results/`,
            image.name,
            `${esrganPath}/LR/`
        ).catch((error) => {
            console.log(error);
            return job.message.reply(
                'Sorry, there was an error processing your image. [me]'
            ).then(() => {
                processNext();
            });
        });
    }

    // Montages the LR and result if argument provided
    if (job.montage && !job.split) {
        console.log('Montaging...');
        await montage(image, job.model, job.message).catch((error) => {
            console.log(error);
            return job.message.reply(
                'Sorry, there was an error processing your image. [mo]'
            ).then(() => {
                processNext();
            });
        });
    }

    // Optimizes the images
    console.log('Optimizing...');
    await optimize(`${esrganPath}/results/`).catch((error) => {
        console.log(error);
        return job.message.reply(
            'Sorry, there was an error processing your image. [o]'
        ).then(() => {
            processNext();
        });
    });

    // Adds all files in results to an array which it will use to send attachments
    let fileNames = fs.readdirSync(`${esrganPath}/results/`);
    let resultImage;
    let montageImage;
    for (let file of fileNames) {
        if (file.includes(job.image.split('.')[0])) {
            resultImage = `${esrganPath}/results/${file}`;
        }
        if (file.includes('_montage')) {
            montageImage = `${esrganPath}/results/${file}`;
        }
    }

    console.log('Sending...');
    job.message
        .reply(`Upscaled using ${job.model}`, {
            // files: [
            //     `${esrganPath}/results/${job.image.split('.')[0]}_rlt.${format}`
            // ]
            files: [resultImage]
        })
        .then(() => {
            if (job.montage) {
                job.message.channel
                    .send(
                        `${job.message.author}, here is the montage you requested`, {
                            files: [montageImage]
                        }
                    )
                    .then(() => {
                        processNext();
                    });
            } else {
                processNext();
            }
        })
        .catch(console.error);
}

function processNext() {
    emptyDirs();
    queue.get(0).jobs.shift();
    try {
        if (queue.get(0).jobs.length > 0) {
            process(queue.get(0).jobs[0]);
        } else {
            queue.delete(0);
        }
    } catch (err) {
        console.log(err);
        queue.delete(0);
    }
}

// Runs ESRGAN
function upscale(model) {
    return new Promise((resolve, reject) => {
        let args = {
            args: [
                `${esrganPath}/models/${model}`,
                `--input=${esrganPath}/LR/`,
                `--output=${esrganPath}/results/`
            ]
        };
        PythonShell.run(esrganPath + '/test.py', args, (err, results) => {
            if (err) {
                // console.log(err);
                // queue.delete(0);
                // return message.channel.send(
                //     'Sorry, there was an error processing your image.'
                // );
                reject(err)
            }

            fs.readdir(`${esrganPath}/results/`, function (err, files) {
                if (err) {
                    // message.channel.send(
                    //     'Sorry, there was an error processing your image.'
                    // );
                    reject(err);
                } else {
                    if (!files.length) {
                        // message.channel.send(
                        //     'Sorry, there was an error processing your image.'
                        // );
                        reject(err);
                    } else {
                        resolve();
                    }
                }
            });
        });
    });
}

// Creates a side-by-side montage
function montage(image, model) {
    let lr = image.path;
    let result = `${esrganPath}results/${image.name}_rlt.png`;
    let modelName = model.replace('.pth', '');

    let path = require('path');
    let absolutePath = path.resolve('./scripts/montage.sh');

    return new Promise((resolve, reject) => {
        shell.exec(
            `${absolutePath} -if="${lr}" -is="${result}" -tf="LR" -ts="${modelName}" -td="2x1" -ug="100%" -io="${image.name}_montage.png" -of="${esrganPath}/results" -f="./scripts/Rubik-Bold.ttf"`, {
                silent: true
            },
            (error, stdout, stderr) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(stdout ? stdout : stderr);
                }
            }
        );
    });
}

// Optimizes output images
function optimize(dir) {
    const imagemin = require('imagemin');
    const imageminOptipng = require('imagemin-optipng');

    return new Promise((resolve, reject) => {
        (async () => {
            await imagemin([dir], {
                use: [imageminOptipng()]
            });

            // Encodes any images that are still over 8mb to lossless webp
            for (let file of fs.readdirSync(dir)) {
                if (!checkUnderSize(dir + file, 8)) {
                    await webpLossless(dir + file);
                }
            }

            // Encodes any images that are still over 8mb to lossy webp
            for (let file of fs.readdirSync(dir)) {
                if (!checkUnderSize(dir + file, 8)) {
                    await webpLossy(dir + file);
                }
            }

            resolve();
        })();
    });
}

// Checks to make sure an image can be sent over Discord
function checkUnderSize(image, size) {
    let stats = fs.statSync(image);
    let fileSizeInBytes = stats['size'];
    let fileSizeInMegabytes = fileSizeInBytes / 1000000.0;
    return fileSizeInMegabytes < size;
}

// Converts to lossless webp using ImageMagick
function webpLossless(image) {
    return new Promise((resolve, reject) => {
        let webpName = image.replace('.png', '.webp');
        shell.exec(
            `magick ${image} -quality 50 -define webp:lossless=true -define webp:target-size:8000000 ${webpName}`,
            (error, stdout, stderr) => {
                if (error) {
                    console.log(error);
                    reject();
                }
                shell.rm('-f', image);
                resolve();
            }
        );
    });
}

// Converts to lossy webp using ImageMagick
function webpLossy(image) {
    return new Promise((resolve, reject) => {
        let webpName = image.replace('.png', '.webp');
        let removePNG = image.includes('.png') ? true : false;
        shell.exec(
            `magick ${image} -quality 75 -define webp:lossless=false -define webp:target-size:8000000 -define webp:pass=4 ${webpName}`,
            (error, stdout, stderr) => {
                if (error) {
                    console.log(error);
                    reject();
                }
                removePNG ? shell.rm('-f', image) : resolve();
                resolve();
            }
        );
    });
}