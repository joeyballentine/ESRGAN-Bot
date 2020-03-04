// TODO:
// Fix bug where upscaling a montage and montaging it send the results twice

// Discord.js import
const Discord = require('discord.js');
const client = new Discord.Client();

const request = require(`request`);
const { downloadModel, downloadImage } = require('./download.js');
const { downscale, convertToPNG, split, merge } = require('./imageUtils.js');

// File stuff
const fs = require(`fs`);
const fsExtra = require('fs-extra');
const sizeOf = require('image-size');
const isImageUrl = require('is-image-url');
const FuzzyMatching = require('fuzzy-matching');
const parsePath = require('parse-filepath');
const path = require('path');

// Shell stuff
const { PythonShell } = require('python-shell');
const shell = require('shelljs');

// The image upscale queue
const queue = new Map();

// Configuration
const { token, prefix, pixelLimit, esrganPath } = require('./config.json');

var fuzzymodels, aliases;

function buildAliases() {
    aliases = {};
    let models = fs.readdirSync(`${esrganPath}/models/`);

    // Create aliases for models based on unique parts
    for (let model of models) {
        let name = path.basename(model, '.pth');
        let parts = name.match(/([0-9]+x?|[A-Z]+(?![a-z])|[A-Z][^A-Z0-9_-]*)/g);
        for (let i = 0; i < parts.length; i++) {
            for (let j = i+1; j <= parts.length; j++) {
                let alias = parts.slice(i, j).join('');
                if (aliases[alias] === undefined) {
                    aliases[alias] = model;
                } else {
                    aliases[alias] = false;
                }
            }
        }
    }

    // Ensure exact names are usable
    for (let model of models) {
        let name = path.basename(model, '.pth');
        aliases[name] = model;
    }

    // Build list of usable aliases
    let fuzzylist = [];
    for (let alias in aliases) {
        if (aliases[alias]) {
            fuzzylist.push(alias);
        }
    }
    console.log('Made ' + fuzzylist.length + ' aliases for ' + models.length + ' models.');
    fuzzymodels = new FuzzyMatching(fuzzylist);
}
buildAliases();

client.on('ready', () => {
    console.log(`Logged in as ${client.user.tag}!`);
    setStatus('Ready to upscale!');
    emptyDirs();
});

client.on('error', console.error);

client.on('message', async message => {
    if (
        !message.content.startsWith(prefix) ||
        message.author.bot ||
        message.channel.type == 'dm'
    )
        return;
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

        // Downloads the image
        let image = url.split('/').pop();

        // Gets the model name from the model argument
        let model = aliases[fuzzymodels.get(
            args[0].endsWith('.pth') ? args[0].slice(0, -4) : args[0]
        ).value];

        // Checks to make sure model name is valid (exists and is spelled right)
        if (!fs.readdirSync(esrganPath + '/models/').includes(model)) {
            buildAliases();
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
                jobs: []
            };
            queue.set(0, queueConstruct);
            queue.get(0).jobs.push(upscaleJob);

            // Process queue until empty
            while (queue.get(0).jobs.length > 0) {
                let currentJob = queue.get(0).jobs[0];
                try {
                    if (queue.get(0).jobs.length > 0) {
                        currentJob.message.channel.send(
                            `${currentJob.image} is being processed using ${currentJob.model}.`
                        );
                        let messages = await process(
                            queue.get(0).jobs[0]
                        ).catch(error => {
                            console.log(error);
                            queue.delete(0);
                            return currentJob.message.reply(error);
                        });
                        console.log('Sending...');
                        setStatus('Sending...');
                        for (let msg of messages) {
                            await currentJob.message
                                .reply(msg.message, {
                                    files: msg.files
                                })
                                .catch(error => {
                                    console.log(error);
                                    queue.delete(0);
                                    return currentJob.message.reply(error);
                                });
                        }
                        setStatus('Ready to upscale!');
                        console.log('Finished processing.');
                    } else {
                        queue.delete(0);
                    }
                } catch (err) {
                    console.log(err);
                    queue.delete(0);
                    throw err;
                }
                emptyDirs();
                queue.get(0).jobs.shift();
            }
            queue.delete(0);
        } else {
            queue.get(0).jobs.push(upscaleJob);
            return message.channel.send(
                `${image} has been added to the queue! Your image is #${queue.get(
                    0
                ).jobs.length - 1} in line for processing.`
            );
        }
    } else if (command === 'models') {
        let files = fs.readdirSync(`${esrganPath}/models/`);
        let table = require('markdown-table');
        let models = [];
        for (let i = 0; i < files.length; i = i + 4) {
            models.push(files.slice(i, i + 4));
        }
        while (models.length > 0) {
            message.channel.send(
                '```' +
                    table(models.splice(0, 16), {
                        rule: false
                    }) +
                    '```'
            );
        }
    } else if (command === 'add') {
        let modelName = args[1].includes('.pth') ? args[1] : args[1] + '.pth';
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
            buildAliases();
            return message.channel.send(
                `Model ${modelName} successfully added.`
            );
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
    // Downloads the image
    let downloaded = await downloadImage(job.url, esrganPath + '/LR/').catch(
        error => {
            console.log(error);
            throw `Sorry, your image failed to download.`;
        }
    );

    // Parses the filename from the downloaded image
    let image = parsePath(downloaded);

    // Ensures a valid image type
    if (!['.png', '.jpg', '.jpeg'].includes(image.ext.toLowerCase())) {
        throw `Sorry, that image cannot be processed.`;
    }

    // Converts image to png if it isn't
    if (image.ext.toLowerCase() !== '.png') {
        await convertToPNG(image.path)
            .catch(error => {
                console.log(error);
                throw 'Sorry, there was an error processing your image. [c]';
            })
            .then(() => {
                image = parsePath(image.dir + '/' + image.name + '.png');
            });
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
        setStatus('Downscaling...');
        await downscale(image.path, job.downscale, job.filter).catch(error => {
            console.log(error);
            throw 'Sorry, there was an error processing your image. [d]';
        });
    }

    // Splits if needed
    if (job.split) {
        console.log('Splitting...');
        setStatus('Splitting...');
        await split(image.path).catch(error => {
            console.log(error);
            throw 'Sorry, there was an error processing your image. [s]';
        });
    }

    // Upscales the image(s)
    console.log('Upscaling...');
    setStatus('Upscaling...');
    await upscale(job.model).catch(error => {
        console.log(error);
        throw 'Sorry, there was an error processing your image. [u]';
    });

    // Merges the images if split was needed
    if (job.split) {
        console.log('Merging...');
        setStatus('Merging...');
        await merge(
            `${esrganPath}/results/`,
            image.name,
            `${esrganPath}/LR/`
        ).catch(error => {
            console.log(error);
            throw 'Sorry, there was an error processing your image. [me]';
        });
    }

    // Montages the LR and result if argument provided
    if (job.montage && !job.split) {
        console.log('Montaging...');
        setStatus('Montaging...');
        await montage(image, job.model, job.message).catch(error => {
            console.log(error);
            throw 'Sorry, there was an error processing your image. [mo]';
        });
    }

    // Optimizes the images
    console.log('Optimizing...');
    setStatus('Optimizing...');
    await optimize(`${esrganPath}/results/`).catch(error => {
        console.log(error);
        throw 'Sorry, there was an error processing your image. [o]';
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
    if (!resultImage) {
        throw 'Sorry, there was an error processing your image. [s]';
    }
    let messages = [];
    messages.push({
        message: `Upscaled using ${job.model}`,
        files: [resultImage]
    });
    if (job.montage) {
        messages.push({
            message: `Here is the montage you requested`,
            files: [montageImage]
        });
    }
    return messages;
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
                reject(err);
            }

            fs.readdir(`${esrganPath}/results/`, function(err, files) {
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
            `${absolutePath} -if="${lr}" -is="${result}" -tf="LR" -ts="${modelName}" -td="2x1" -ug="100%" -io="${image.name}_montage.png" -of="${esrganPath}/results" -f="./scripts/Rubik-Bold.ttf"`,
            {
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

function setStatus(status) {
    client.user
        .setActivity(status, { type: 'PLAYING' })
        // .then(presence =>
        //     console.log(
        //         `Activity set to ${presence.game ? presence.game.name : 'none'}`
        //     )
        // )
        .catch(console.error);
}
