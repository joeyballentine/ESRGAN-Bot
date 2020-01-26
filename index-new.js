// Discord.js import
const Discord = require('discord.js');
const client = new Discord.Client();

const request = require(`request`);

// File stuff
const fs = require(`fs`);
const fsExtra = require('fs-extra');

// Image downloading stuff
const isImageUrl = require('is-image-url');
// const download = require('image-downloader');

// Python shell
const { PythonShell } = require('python-shell');
const shell = require('shelljs');

// The image upscale queue
const queue = new Map();

// The prefix used for commands
const prefix = '!';

// Change these depending on what you want to allow
const pixelLimit = 500 * 500;
const sizeLimit = 500000;

// Path to ESRGAN. Should be initialized by a submodule and is meant to be used with BlueAmulet's fork
const esrganPath = './ESRGAN/';

// Connects to the bot account and empties the directories
client.on('ready', () => {
    console.log(`Logged in as ${client.user.tag}!`);
    emptyDirs();
});

// Message event handler
client.on('message', async message => {
    // Removes extra spaces between commands
    // this doesnt work
    message.content = message.content.replace(/ +(?= )/g, '');

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
        await download(url, image);

        // Gets the model name from the model argument
        let model = args[0].includes('.pth') ? args[0] : args[0] + '.pth';

        // Checks to make sure model name is valid (exists and is spelled right)
        if (!fs.readdirSync(esrganPath + '/models/').includes(model)) {
            return message.channel.send('Not a valid model.');
        }

        // The job sent for processing
        let upscaleJob = {
            model: model,
            image: image,
            downscale: false,
            filter: 'box',
            montage: false,
            message: message
        };

        // Parsing the extra arguments

        // downscale
        //upscaleJob.downscale = args[args.indexOf(arg) + 1];
        // upscaleJob.downscale = ['--downscale', '-r'].some(arg => {
        //     console.log('downscale!')
        //     if (args.includes(arg)) {
        //         console.log(args[args.indexOf(arg) + 1])
        //         return args[args.indexOf(arg) + 1]
        //     };
        // })

        if (args.includes('-downscale')) {
            upscaleJob.downscale = args[args.indexOf('-downscale') + 1];
        }
        //console.log(upscaleJob.downscale);

        // filter
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
                    `${image} has been added to the queue! Your image is #${queue.length} in line for processing.`
                );
            }
        } else {
            return message.channel.send(
                `Sorry, that image cannot be processed.`
            );
        }
    }
});

client.login('NjYzMTA3NTQ3OTg4NzU0NDUy.XhDtGg.CGxZaTJRr7OmYJOVbBlY2j9bspc');

function emptyDirs() {
    fsExtra.emptyDirSync(esrganPath + '/results/');
    fsExtra.emptyDirSync(esrganPath + '/LR/');
}

function download(url, image) {
    return new Promise((resolve, reject) => {
        request
            .get(url)
            .on('error', console.error)
            .pipe(fs.createWriteStream(esrganPath + '/LR/' + image))
            .on('finish', () => {
                console.log(`The file is finished downloading.`);
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
    if (job.downscale) await downscale(job.image, job.downscale, job.filter);
    //split();
    await upscale(job.image, job.model);
    //merge();
    optimize();

    if (job.montage) await montage(job.image, job.model, job.message);

    return job.message
        .reply(`Upscaled using ${job.model}`, {
            files: [`${esrganPath}/results/${job.image.split('.')[0]}_rlt.png`]
        })
        .then(() => {
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

function upscale(image, model) {
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

            try {
                if (!fs.existsSync(filePath)) {
                    return message.channel.send(
                        'Sorry, there was an error processing your image.'
                    );
                } else {
                    resolve();
                }
            } catch (err) {
                return message.channel.send(
                    'Sorry, there was an error processing your image.'
                );
            }
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
    let result = `${esrganPath}results/${image.split('.')[0]}_rlt.png`;
    let modelName = model.replace('.pth', '');

    let path = require('path');
    let absolutePath = path.resolve('./scripts/montage.sh');

    return new Promise((resolve, reject) => {
        shell.rm('-rf', '/montages/');
        shell.exec(
            `${absolutePath} -if="${lr}" -is="${result}" -tf="LR" -ts="${modelName}" -td="2x1" -ug="100%" -io="output_montage.png"`,
            (error, stdout, stderr) => {
                if (error) {
                    console.warn(error);
                    message.channel.send(
                        'There was an error making your montage.'
                    );
                } else {
                    message.channel.send('', {
                        files: [`./montages/output_montage.png`]
                    });
                }
                resolve(stdout ? stdout : stderr);
            }
        );
    });
}

function split() {
    shell.exec(`./scripts/split.sh`);
}

function merge() {
    shell.exec(`./scripts/merge.sh`);
}

function optimize() {
    const imagemin = require('imagemin');
    const imageminOptipng = require('imagemin-optipng');

    (async () => {
        await imagemin(
            [`${esrganPath}/results/*.png`],
            `${esrganPath}/results/`,
            {
                use: [imageminOptipng()]
            }
        );

        console.log('Images optimized!');
    })();
}
