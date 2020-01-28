// Discord.js import
const Discord = require('discord.js');
const client = new Discord.Client();

const request = require(`request`);

// File stuff
const fs = require(`fs`);
const fsExtra = require('fs-extra');
const sizeOf = require('image-size');
const isImageUrl = require('is-image-url');

// Shell stuff
const { PythonShell } = require('python-shell');
const shell = require('shelljs');

// The image upscale queue
const queue = new Map();

// Configuration
const { token, prefix, pixelLimit, esrganPath } = require('./config.json');

// Connects to the bot account and empties the directories
client.on('ready', () => {
    console.log(`Logged in as ${client.user.tag}!`);
    emptyDirs();
});

// Message event handler
client.on('message', async message => {
    // The bot will not respond unless the prefix is typed
    // It will also ignore anything sent by itself
    if (!message.content.startsWith(prefix) || message.author.bot) return;

    // Splits the args into an array
    const args = message.content.slice(prefix.length).split(' ');

    // Strips the command off of the args array
    const command = args.shift().toLowerCase();

    // Does all the steps required for upscaling the image
    if (command === 'upscale') {
        // If no args are given the bot will stop and send an error message
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
        url = url.split('&')[0];
        url = url.split('?')[0];

        // Downloads the image
        let image = url.split('/').pop();

        // Gets the model name from the model argument
        let model = args[0].includes('.pth') ? args[0] : args[0] + '.pth';

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

        // Checks if the image is valid to be upscaled
        if (checkImage(image)) {
            // Adds to the queue and starts upscaling if not already started.
            if (!queue.get(0)) {
                const queueConstruct = {
                    textChannel: message.channel,
                    jobs: []
                };
                queue.set(0, queueConstruct);
                queue.get(0).jobs.push(upscaleJob);

                try {
                    message.channel.send(`Your image is being processed.`);
                    process(queue.get(0).jobs[0]);
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
        } else {
            return message.channel.send(
                `Sorry, that image cannot be processed.`
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
        //return message.channel.send(files);
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
        download(args[0], `${esrganPath}/models/${modelName}`);
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

\`-filter [imagemagick filter]\` // Filter to be used for downscaling. Must be vlaid imagemagick filter. Defaults to box.

\`-montage\` // Creates aside by side comparison of the LR and result after upscaling

Example: \`${prefix}upscale www.imageurl.com/image.png 4xBox.pth -downscale 4 -filter point -montage\``;
        return message.channel.send(help);
    }
});

client.login(token);

function emptyDirs() {
    fsExtra.emptyDirSync(esrganPath + '/results/');
    fsExtra.emptyDirSync(esrganPath + '/LR/');
}

function download(url, destination) {
    return new Promise((resolve, reject) => {
        request
            .get(url)
            .on('error', console.error)
            .pipe(fs.createWriteStream(destination))
            .on('finish', () => {
                // console.log(`The file is finished downloading.`);
                resolve();
            })
            .on('error', error => {
                reject(error);
            });
    });
}

function checkImage(image) {
    if (
        ['png', 'jpg', 'jpeg'].some(
            filetype => image.split('.').pop() === filetype.toLowerCase()
        )
    ) {
        return true;
    } else return false;
}

async function process(job) {
    // Downloads the image
    await download(job.url, esrganPath + '/LR/' + job.image);

    // Checks image to see if it should split
    let dimensions = sizeOf(esrganPath + '/LR/' + job.image);
    if (job.downscale) {
        dimensions.width /= job.downscale;
        dimensions.height /= job.downscale;
    }
    if (dimensions.width >= pixelLimit || dimensions.height >= pixelLimit) {
        job.split = true;
    }

    // Downscales the image if argument provided
    if (job.downscale) await downscale(job.image, job.downscale, job.filter);

    // Splits if needed
    if (job.split) await split();

    // Upscales the image(s)
    await upscale(job.image, job.model, job.message);

    // Merges the images if split was needed
    if (job.split) await merge(job.image);

    // Optimizes the images
    await optimize(job.image);

    // Montages the LR and result if argument provided
    if (job.montage && !job.split)
        await montage(job.image, job.model, job.message);

    // Adds all files in results to an array which it will use to send attachments
    let fileNames = fs.readdirSync(`${esrganPath}/results/`);
    let files = [];
    for (let file of fileNames) {
        files.push(`${esrganPath}/results/${file}`);
    }

    return job.message
        .reply(`Upscaled using ${job.model}`, {
            // files: [
            //     `${esrganPath}/results/${job.image.split('.')[0]}_rlt.${format}`
            // ]
            files: files
        })
        .then(() => {
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
                return job.message.channel.send(err);
            }
        });
}

function upscale(image, model, message) {
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
                console.log(err);
                queue.delete(0);
                return message.channel.send(
                    'Sorry, there was an error processing your image.'
                );
            }

            let filePath = `${esrganPath}/results/${
                image.split('.')[0]
            }_rlt.png`;

            fs.readdir(`${esrganPath}/results/`, function(err, files) {
                if (err) {
                    return message.channel.send(
                        'Sorry, there was an error processing your image.'
                    );
                } else {
                    if (!files.length) {
                        return message.channel.send(
                            'Sorry, there was an error processing your image.'
                        );
                    } else {
                        resolve();
                    }
                }
            });
        });
    });
}

function downscale(image, amount, filter) {
    return new Promise((resolve, reject) => {
        shell.exec(
            `magick mogrify -resize ${(1.0 / amount) * 100.0 +
                '%'} -filter ${filter} ${esrganPath}LR/*.*`,
            (error, stdout, stderr) => {
                if (error) {
                    console.warn(error);
                }
                resolve(stdout ? stdout : stderr);
            }
        );
    });
}

function montage(image, model, message) {
    //TODO extract image % difference for scaling
    let lr = `${esrganPath}LR/${image}`;
    let imageName = image.split('.')[0];
    let result = `${esrganPath}results/${imageName}_rlt.png`;
    let modelName = model.replace('.pth', '');

    let path = require('path');
    let absolutePath = path.resolve('./scripts/montage.sh');

    return new Promise((resolve, reject) => {
        //shell.rm('-rf', '/montages/');
        shell.exec(
            `${absolutePath} -if="${lr}" -is="${result}" -tf="LR" -ts="${modelName}" -td="2x1" -ug="100%" -io="${imageName}_montage.png" -of="${esrganPath}/results"`,
            { silent: true },
            (error, stdout, stderr) => {
                if (error) {
                    console.warn(error);
                    message.channel.send(
                        'There was an error making your montage.'
                    );
                } else {
                    shell.exec(
                        `magick ${esrganPath}/results/${imageName}_montage.png -quality 50 -define webp:target-size=8000000 ${esrganPath}/results/${imageName}_montage.webp`,
                        () => {
                            shell.rm(
                                '-f',
                                `${esrganPath}/results/${imageName}_montage.png`
                            );
                            resolve(stdout ? stdout : stderr);
                        }
                    );
                }
                //resolve(stdout ? stdout : stderr);
            }
        );
    });
}

function split() {
    let path = require('path');
    let absolutePath = path.resolve('./scripts/split.sh');

    return new Promise((resolve, reject) => {
        shell.exec(
            `${absolutePath} ${esrganPath}/LR`,
            (error, stdout, stderr) => {
                if (error) console.log(error);
                resolve();
            }
        );
    });
}

function merge(image) {
    let path = require('path');
    let absolutePath = path.resolve('./scripts/merge.sh');

    let imageName = image.split('.')[0];
    return new Promise((resolve, reject) => {
        shell.exec(
            `${absolutePath} ${esrganPath}/LR ${esrganPath}/results ${imageName}`,
            (error, stdout, stderr) => {
                if (error) console.log(error);
                resolve();
            }
        );
    });
}

function optimize(image) {
    const imagemin = require('imagemin');
    const imageminOptipng = require('imagemin-optipng');

    let imageName = image.split('.')[0];

    return new Promise((resolve, reject) => {
        (async () => {
            await imagemin([`${esrganPath}/results/${imageName}_rlt.png`], {
                use: [imageminOptipng()]
            });

            //console.log('Images optimized!');

            // Checks filesize for Discord's limits and converts to lossy webp if >= 8mb
            let stats = fs.statSync(
                `${esrganPath}/results/${imageName}_rlt.png`
            );
            let fileSizeInBytes = stats['size'];
            let fileSizeInMegabytes = fileSizeInBytes / 1000000.0;
            if (fileSizeInMegabytes >= 8) {
                shell.exec(
                    `magick ${esrganPath}/results/${imageName}_rlt.png -quality 50 -define webp:target-size=8000000 ${esrganPath}/results/${imageName}_rlt.webp`,
                    (error, stdout, stderr) => {
                        shell.rm(
                            '-f',
                            `${esrganPath}/results/${imageName}_rlt.png`
                        );
                        resolve();
                    }
                );
            } else {
                resolve();
            }
        })();
    });
}
